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
from PIL import Image, ImageDraw

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


def render_grid_node(state: BeadState, cell_px: Optional[int] = None) -> BeadState:
    """
    格子線付きのアイロンビーズ図面PNGを描画する。
    cell_px を指定しない場合は、マス数に応じて自動でサイズ調整される
    （50x50のような大きいグリッドでも見やすく、崩れないようにするため）。
    """
    if state.get("error"):
        return state
    try:
        idx_map = state["color_map"]
        color_db = state["color_db"]
        h, w = idx_map.shape

        px = cell_px if cell_px is not None else _auto_cell_px(w, h)
        # セルが小さいほどビーズの縁取り(outline)を細く/省略して潰れを防ぐ
        outline = (80, 80, 80) if px >= 10 else None
        inset = 2 if px >= 14 else (1 if px >= 8 else 0)

        img_w = w * px + 1
        img_h = h * px + 1
        canvas = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(canvas)

        for y in range(h):
            for x in range(w):
                c = color_db[int(idx_map[y, x])]
                rgb = (c["r"], c["g"], c["b"])
                x0, y0 = x * px, y * px
                x1, y1 = x0 + px, y0 + px
                draw.ellipse(
                    [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                    fill=rgb, outline=outline
                )

        # 格子線（グリッド）
        for x in range(w + 1):
            xp = x * px
            draw.line([(xp, 0), (xp, img_h)], fill=(200, 200, 200), width=1)
        for y in range(h + 1):
            yp = y * px
            draw.line([(0, yp), (img_w, yp)], fill=(200, 200, 200), width=1)

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
