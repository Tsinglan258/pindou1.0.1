"""
agent_bead.py
=================================================
アイロンビーズ変換エージェント（LangGraph StateGraph版）

画像 → ピクセル化 → Artkalビーズ色マッチング → 個数集計 → 格子図面PNG生成
という一連の処理を、LangGraphのStateGraphでノードとして順番に実行する。

LLMは使用しない。StateGraphは「ツール関数の実行順序をスケジュールする
だけの決定的なパイプライン」として利用している。

拡張ポイント（後から機能追加しやすいように、あえて分離してある）:
  - 手動で1マスだけ色を修正したい場合
        -> color_map (2次元配列) の該当セルを書き換えて
           render_grid_node() だけを呼び直せばよい。
  - Hama色などブランド切り替えをしたい場合
        -> color_db を差し替えるだけで動く（load_color_db関数を参照）。
  - テキスト出力をダウンロードさせたい場合
        -> build_result_text() の戻り値をそのままダウンロードボタンに渡せる。
=================================================
"""

from __future__ import annotations

import json
import io
from typing import TypedDict, Optional, List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from langgraph.graph import StateGraph, START, END


# =================================================
# 状態定義（LangGraphのState）
# =================================================
class BeadState(TypedDict, total=False):
    # 入力
    image_bytes: bytes
    target_width: int
    target_height: int
    max_colors: Optional[int]        # 使用色数の上限（Noneなら無制限）
    color_db_path: str               # artkal_color.json のパス
    color_db: List[Dict[str, Any]]   # 読み込み済み色データベース

    # 中間生成物
    pixel_grid: Any                  # np.ndarray (H, W, 3) 縮小後のRGB
    color_map: Any                   # np.ndarray (H, W) 各セルのcolor_db内インデックス

    # 出力
    color_counts: Dict[str, int]     # {color_code: 個数}
    output_image: Any                # PIL.Image 格子図面
    result_text: str                 # 色番号と個数の一覧テキスト

    # エラー
    error: Optional[str]


# =================================================
# 各ノード（ツール関数）
# =================================================
def load_color_db_node(state: BeadState) -> BeadState:
    """artkal_color.json を読み込み、state["color_db"] にセットする。"""
    if state.get("error"):
        return state
    try:
        path = state.get("color_db_path", "artkal_color.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        colors = data.get("colors", data if isinstance(data, list) else [])
        if not colors:
            raise ValueError("色データベースが空です。")
        state["color_db"] = colors
    except Exception as e:
        state["error"] = f"色データベースの読み込みに失敗しました: {e}"
    return state


def pixelate_node(state: BeadState) -> BeadState:
    """画像を読み込み、指定サイズへ最近傍補間（NEAREST）で縮小する。"""
    if state.get("error"):
        return state
    try:
        w = int(state["target_width"])
        h = int(state["target_height"])
        if w <= 0 or h <= 0:
            raise ValueError("幅・高さは1以上の整数を指定してください。")
        if w > 300 or h > 300:
            raise ValueError("幅・高さは300以下にしてください（負荷軽減のため）。")

        img = Image.open(io.BytesIO(state["image_bytes"])).convert("RGB")
        # ぼかさず、最近傍補間でジャギー気味に縮小する
        small = img.resize((w, h), resample=Image.NEAREST)
        state["pixel_grid"] = np.array(small, dtype=np.int32)  # (H, W, 3)
    except Exception as e:
        state["error"] = f"画像のピクセル化に失敗しました: {e}"
    return state


def _build_color_array(color_db: List[Dict[str, Any]]) -> np.ndarray:
    """色DBから (K, 3) のRGB配列を作る。"""
    return np.array([[c["r"], c["g"], c["b"]] for c in color_db], dtype=np.int32)


def match_colors_node(state: BeadState) -> BeadState:
    """
    各ピクセルのRGBに対し、ユークリッド距離で最も近いArtkal色を割り当てる。
    max_colors が指定されている場合は、
      1) 一旦フルパレットでマッチングして使用頻度を数え、
      2) 頻度上位 max_colors 色だけを残したサブパレットで再マッチングする。
    という2段階処理で「使用色数の上限」を実現する。
    """
    if state.get("error"):
        return state
    try:
        pixel_grid = state["pixel_grid"]
        color_db = state["color_db"]
        max_colors = state.get("max_colors")

        def match(colors: List[Dict[str, Any]]) -> np.ndarray:
            palette = _build_color_array(colors)  # (K, 3)
            # (H, W, 1, 3) - (1, 1, K, 3) -> (H, W, K, 3)
            diff = pixel_grid[:, :, None, :] - palette[None, None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)  # (H, W, K)
            return np.argmin(dist2, axis=-1)       # (H, W) : colors内インデックス

            
        idx_map = match(color_db)

        if max_colors is not None and max_colors > 0 and max_colors < len(color_db):
            # 使用頻度の高い色だけを残す
            unique, counts = np.unique(idx_map, return_counts=True)
            order = np.argsort(-counts)
            top_indices = unique[order][:max_colors]
            sub_db = [color_db[i] for i in top_indices]
            idx_map_sub = match(sub_db)
            # sub_db 内インデックス -> 元のcolor_dbインデックスへ変換
            remap = np.array(top_indices, dtype=np.int32)
            idx_map = remap[idx_map_sub]

        state["color_map"] = idx_map
    except Exception as e:
        state["error"] = f"色マッチングに失敗しました: {e}"
    return state


def count_colors_node(state: BeadState) -> BeadState:
    """各色コードごとの必要個数を集計する。"""
    if state.get("error"):
        return state
    try:
        idx_map = state["color_map"]
        color_db = state["color_db"]
        unique, counts = np.unique(idx_map, return_counts=True)
        color_counts: Dict[str, int] = {}
        for i, c in zip(unique, counts):
            code = color_db[int(i)]["code"]
            color_counts[code] = int(c)
        # 個数の多い順に並べ替え
        state["color_counts"] = dict(
            sorted(color_counts.items(), key=lambda kv: -kv[1])
        )
    except Exception as e:
        state["error"] = f"個数集計に失敗しました: {e}"
    return state


def _auto_cell_px(w: int, h: int, target_canvas_px: int = 1100,
                   min_px: int = 8, max_px: int = 28) -> int:
    """
    マス数(w, h)に応じて、見やすさと画像サイズのバランスが取れる
    セルサイズ(px)を自動計算する。
    マス数が多い（例:50x50）ほどセルを小さく、少ないほど大きくする。
    """
    longer_side = max(w, h)
    if longer_side <= 0:
        return max_px
    cell_px = target_canvas_px // longer_side
    return int(max(min_px, min(max_px, cell_px)))


# --- フォント関連ヘルパー ---------------------------------------------
# デプロイ先(Streamlit Community Cloud等)でフォントが無くても落ちないよう、
# 複数の候補パスを順番に試し、見つからなければPILの標準フォントにフォールバックする。
# 日本語の色名を綺麗に表示したい場合は、リポジトリのpackages.txtに
# "fonts-noto-cjk" を追記してデプロイすることを推奨（README的な位置づけ）。
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_ASCII_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
] + _CJK_FONT_CANDIDATES

_FONT_CACHE: Dict[Tuple[str, int], Any] = {}


def _get_font(size: int, prefer_cjk: bool = False):
    """日本語表示が必要な箇所は prefer_cjk=True、英数字のみならFalseで呼ぶ。"""
    key = ("cjk" if prefer_cjk else "ascii", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = _CJK_FONT_CANDIDATES if prefer_cjk else _ASCII_FONT_CANDIDATES
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _text_size(draw: "ImageDraw.ImageDraw", text: str, font) -> Tuple[int, int]:
    """Pillowのバージョン差異を吸収してテキストサイズを取得する。"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)  # 古いPillow向けフォールバック


def _col_label(index0: int) -> str:
    """0始まりの列番号をExcel式のアルファベット座標(A, B, ... Z, AA, AB, ...)に変換する。"""
    n = index0 + 1
    label = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(65 + rem) + label
    return label


def render_grid_node(state: BeadState, cell_px: Optional[int] = None) -> BeadState:
    """
    アイロンビーズ図面を1枚のPNGとして描画する。
    含まれる要素（すべて同一画像内）:
      - 上端: 列座標（A, B, C, ...）
      - 左端: 行座標（1, 2, 3, ...）
      - 中央: 格子線付きのビーズ図案本体
      - 下部: 使用色番号一覧（カラーレジェンド：色見本・色番号・色名・個数）
    cell_px を指定しない場合は、マス数に応じて自動でサイズ調整される
    （50x50のような大きいグリッドでも見やすく、崩れないようにするため）。
    """
    if state.get("error"):
        return state
    try:
        idx_map = state["color_map"]
        color_db = state["color_db"]
        color_counts = state.get("color_counts", {})
        h, w = idx_map.shape

        px = cell_px if cell_px is not None else _auto_cell_px(w, h)
        outline = (80, 80, 80) if px >= 10 else None
        inset = 2 if px >= 14 else (1 if px >= 8 else 0)

        grid_w = w * px
        grid_h = h * px

        # --- 座標ラベル用のフォント・余白サイズを決定 ---
        # 列数が多い場合(AA, AB, ...のように2文字になる場合)はセル幅に収まるよう
        # フォントサイズを自動的に縮小し、座標がつぶれて重ならないようにする。
        dummy_img = Image.new("RGB", (10, 10))
        dummy_draw = ImageDraw.Draw(dummy_img)
        max_col_label_text = _col_label(w - 1) if w > 0 else "A"

        label_font_size = int(max(9, min(px - 4, 16)))
        label_font = _get_font(label_font_size, prefer_cjk=False)
        while label_font_size > 6:
            col_label_w = _text_size(dummy_draw, max_col_label_text, label_font)[0]
            if col_label_w <= px - 2:
                break
            label_font_size -= 1
            label_font = _get_font(label_font_size, prefer_cjk=False)

        max_row_label_w = max(
            (_text_size(dummy_draw, str(y + 1), label_font)[0] for y in range(h)),
            default=10,
        )
        left_margin = max(24, max_row_label_w + 10)
        top_margin = max(22, label_font_size + 12)
        right_margin = 12
        top_pad = 10  # 一番上の余白

        # --- レジェンド（色番号一覧）のレイアウトを決定 ---
        legend_font_size = 13
        legend_title_font = _get_font(legend_font_size + 2, prefer_cjk=True)
        legend_font = _get_font(legend_font_size, prefer_cjk=True)
        swatch_size = 16
        legend_item_h = 24
        legend_gap_x = 14

        legend_items = []  # (rgb, label_text)
        for code, count in color_counts.items():
            c = next((cc for cc in color_db if cc["code"] == code), None)
            if c is None:
                continue
            label = f"{code} {c['name']} x{count}"
            legend_items.append(((c["r"], c["g"], c["b"]), label))

        total_canvas_w = max(grid_w + left_margin + right_margin, 480)

        # レジェンド1件あたりの表示幅をテキスト計測して決め、折り返し列数を算出
        max_label_w = max(
            (_text_size(dummy_draw, lbl, legend_font)[0] for _, lbl in legend_items),
            default=80,
        )
        legend_item_w = swatch_size + 6 + max_label_w + legend_gap_x
        legend_cols = max(1, int((total_canvas_w - 24) // legend_item_w))
        legend_rows = (len(legend_items) + legend_cols - 1) // max(legend_cols, 1) if legend_items else 0

        legend_title_h = getattr(legend_title_font, "size", legend_font_size + 2) + 12
        legend_top_pad = 14
        legend_h = legend_top_pad + legend_title_h + legend_rows * legend_item_h + 14

        img_w = int(total_canvas_w) + 1
        img_h = int(top_pad + top_margin + grid_h + legend_h) + 1

        canvas = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(canvas)

        origin_x = left_margin
        origin_y = top_pad + top_margin

        # --- 列座標（アルファベット）を上端に描画 ---
        for x in range(w):
            label = _col_label(x)
            tw, th = _text_size(draw, label, label_font)
            cx = origin_x + x * px + px / 2 - tw / 2
            cy = top_pad + (top_margin - th) / 2
            draw.text((cx, cy), label, fill=(40, 40, 40), font=label_font)

        # --- 行座標（数字）を左端に描画 ---
        for y in range(h):
            label = str(y + 1)
            tw, th = _text_size(draw, label, label_font)
            cx = origin_x - tw - 6
            cy = origin_y + y * px + px / 2 - th / 2
            draw.text((cx, cy), label, fill=(40, 40, 40), font=label_font)

        # --- ビーズ本体（ドット）を描画 ---
        for y in range(h):
            for x in range(w):
                c = color_db[int(idx_map[y, x])]
                rgb = (c["r"], c["g"], c["b"])
                x0, y0 = origin_x + x * px, origin_y + y * px
                x1, y1 = x0 + px, y0 + px
                draw.ellipse(
                    [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                    fill=rgb, outline=outline
                )

        # --- 格子線 ---
        for x in range(w + 1):
            xp = origin_x + x * px
            draw.line([(xp, origin_y), (xp, origin_y + grid_h)], fill=(200, 200, 200), width=1)
        for y in range(h + 1):
            yp = origin_y + y * px
            draw.line([(origin_x, yp), (origin_x + grid_w, yp)], fill=(200, 200, 200), width=1)

        # 図案全体を囲む枠線
        draw.rectangle(
            [origin_x, origin_y, origin_x + grid_w, origin_y + grid_h],
            outline=(120, 120, 120), width=1
        )

        # --- 使用色番号一覧（レジェンド）を下部に描画 ---
        legend_top = origin_y + grid_h + legend_top_pad
        draw.line(
            [(12, legend_top - 6), (img_w - 12, legend_top - 6)],
            fill=(220, 220, 220), width=1
        )
        draw.text((12, legend_top), "色番号一覧 / 色号列表 (Color Legend)",
                   fill=(20, 20, 20), font=legend_title_font)

        legend_start_y = legend_top + legend_title_h
        for i, (rgb, label) in enumerate(legend_items):
            col = i % legend_cols
            row = i // legend_cols
            ix = 12 + col * legend_item_w
            iy = legend_start_y + row * legend_item_h
            draw.ellipse(
                [ix, iy + 2, ix + swatch_size, iy + 2 + swatch_size],
                fill=rgb, outline=(90, 90, 90)
            )
            draw.text((ix + swatch_size + 6, iy), label, fill=(20, 20, 20), font=legend_font)

        state["output_image"] = canvas
    except Exception as e:
        state["error"] = f"図面描画に失敗しました: {e}"
    return state


def build_result_text_node(state: BeadState) -> BeadState:
    """色コードと必要個数の一覧をテキスト化する（ダウンロード機能等に利用可）。"""
    if state.get("error"):
        return state
    try:
        color_db = {c["code"]: c for c in state["color_db"]}
        lines = ["色番号\t色名\t必要個数"]
        total = 0
        for code, count in state["color_counts"].items():
            name = color_db.get(code, {}).get("name", "")
            lines.append(f"{code}\t{name}\t{count}")
            total += count
        lines.append(f"---\t合計\t{total}")
        state["result_text"] = "\n".join(lines)
    except Exception as e:
        state["error"] = f"テキスト生成に失敗しました: {e}"
    return state


# =================================================
# グラフ構築
# =================================================
def build_graph():
    graph = StateGraph(BeadState)

    graph.add_node("load_color_db", load_color_db_node)
    graph.add_node("pixelate", pixelate_node)
    graph.add_node("match_colors", match_colors_node)
    graph.add_node("count_colors", count_colors_node)
    graph.add_node("render_grid", render_grid_node)
    graph.add_node("build_result_text", build_result_text_node)

    graph.add_edge(START, "load_color_db")
    graph.add_edge("load_color_db", "pixelate")
    graph.add_edge("pixelate", "match_colors")
    graph.add_edge("match_colors", "count_colors")
    graph.add_edge("count_colors", "render_grid")
    graph.add_edge("render_grid", "build_result_text")
    graph.add_edge("build_result_text", END)

    return graph.compile()


_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


# =================================================
# 外部から呼び出すエントリポイント
# =================================================
def run_agent(
    image_bytes: bytes,
    target_width: int,
    target_height: int,
    max_colors: Optional[int] = None,
    color_db_path: str = "artkal_color.json",
) -> BeadState:
    """
    アイロンビーズ変換エージェントを実行する。

    Returns:
        BeadState: 実行結果を含む最終状態。
                   state["error"] が None でない場合はエラー発生。
                   成功時は state["output_image"] (PIL.Image) と
                   state["color_counts"] (dict) が利用可能。
    """
    initial_state: BeadState = {
        "image_bytes": image_bytes,
        "target_width": target_width,
        "target_height": target_height,
        "max_colors": max_colors,
        "color_db_path": color_db_path,
        "error": None,
    }
    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # 簡易動作確認用（サンプル画像がある場合のみ）
    print("agent_bead.py: run_agent() を app.py もしくは他スクリプトから呼び出してください。")
