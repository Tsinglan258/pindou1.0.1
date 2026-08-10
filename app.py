"""
app.py
=================================================
アイロンビーズ変換ツール - Streamlit フロントエンド

agent_bead.py の run_agent() を呼び出して、
画像アップロード → 図面生成 → 結果表示 までを行う。

【今回の変更点】
  1. 出力画像を1枚のPNGに統合（agent_bead.py側で対応。
     ビーズ図案・座標(A,B,C.../1,2,3...)・色番号一覧(レジェンド)が
     すべて同じ画像内に描画されるようになったため、
     app.py側では st.image を1回呼ぶだけでよい）。
  2. 日本語／中国語の言語切り替え機能を追加（TRANSLATIONS辞書 + t()関数）。
  3. 座標グリッドは画像内(agent_bead.py)で対応済み。

agent_bead.py の読み込み処理・変換ロジック・処理速度には一切手を加えていない。

拡張ポイント:
  - 「1マス手動修正」機能を足す場合は、st.session_state に
    color_map / color_db を保持し、セル選択UI(例: st.columns + selectbox)
    から color_map を書き換え、agent_bead.render_grid_node() /
    agent_bead.count_colors_node() / build_result_text_node() を
    直接呼び直して再描画すればよい（このファイルの一番下にヒントを記載）。
  - 「Hama色に切り替え」機能を足す場合は、color_db_path を
    hama_color.json 等に差し替えるセレクトボックスを追加すればよい。
  - 「テキスト出力ダウンロード」は既に実装済み（st.download_button）。
  - 言語を追加したい場合は TRANSLATIONS 辞書に "en" などのキーを追加するだけでよい。
=================================================
"""

import io
import os

import streamlit as st
from PIL import Image

from agent_bead import run_agent

# -------------------------------------------------
# 多言語対応（日本語 / 中国語）
# -------------------------------------------------
TRANSLATIONS = {
    "ja": {
        "page_title": "アイロンビーズ変換ツール",
        "title": "🧵 アイロンビーズ変換ツール",
        "caption": "画像をアップロードすると、Artkalビーズの色番号に変換した図案を作成します。",
        "settings_header": "⚙️ 設定",
        "uploader_label": "画像をアップロード",
        "width_label": "横幅（マス数）",
        "height_label": "縦幅（マス数）",
        "use_max_colors_label": "使用色数を制限する",
        "max_colors_label": "最大使用色数",
        "run_button": "🚀 変換実行",
        "preview_subheader": "アップロード画像プレビュー",
        "spinner_text": "エージェントが変換中です...",
        "error_no_upload": "先に画像をアップロードしてください。",
        "error_prefix": "エラーが発生しました: ",
        "success_text": "変換が完了しました！",
        "image_subheader": "🖼️ アイロンビーズ図面（座標・色番号一覧つき）",
        "download_png_button": "図面PNGをダウンロード",
        "legend_subheader": "🎨 使用ビーズ一覧",
        "col_code": "色番号",
        "col_name": "色名",
        "col_count": "必要個数",
        "total_text": "合計ビーズ数: {total} 個 / 使用色数: {colors} 色",
        "download_text_button": "色番号・個数リストをテキストでダウンロード",
        "info_initial": "左側のサイドバーから画像をアップロードし、「変換実行」を押してください。",
    },
    "zh": {
        "page_title": "拼豆图案转换工具",
        "title": "🧵 拼豆图案转换工具",
        "caption": "上传图片后，将自动转换为Artkal拼豆色号图案。",
        "settings_header": "⚙️ 设置",
        "uploader_label": "上传图片",
        "width_label": "宽度（格数）",
        "height_label": "高度（格数）",
        "use_max_colors_label": "限制使用颜色数量",
        "max_colors_label": "最大颜色数",
        "run_button": "🚀 开始转换",
        "preview_subheader": "上传图片预览",
        "spinner_text": "正在生成中，请稍候...",
        "error_no_upload": "请先上传图片。",
        "error_prefix": "发生错误：",
        "success_text": "转换完成！",
        "image_subheader": "🖼️ 拼豆图案（含坐标与色号列表）",
        "download_png_button": "下载图案PNG",
        "legend_subheader": "🎨 使用拼豆列表",
        "col_code": "色号",
        "col_name": "颜色名称",
        "col_count": "所需数量",
        "total_text": "拼豆总数：{total} 个 / 使用颜色数：{colors} 种",
        "download_text_button": "下载色号与数量列表（文本）",
        "info_initial": "请在左侧侧边栏上传图片，然后点击“开始转换”。",
    },
}


def t(key: str) -> str:
    """現在の言語設定に応じた翻訳済みテキストを取得する。"""
    lang = st.session_state.get("lang", "ja")
    return TRANSLATIONS.get(lang, TRANSLATIONS["ja"]).get(key, key)


if "lang" not in st.session_state:
    st.session_state["lang"] = "ja"  # 初期表示は日本語

# -------------------------------------------------
# 基本設定
# -------------------------------------------------
st.set_page_config(page_title=t("page_title"), page_icon="🧵", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLOR_DB_PATH = os.path.join(BASE_DIR, "artkal_color.json")

# --- 言語切り替えボタン（画面右上に配置） ---
top_left, top_right = st.columns([5, 1])
with top_right:
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("日本語", use_container_width=True,
                      type="primary" if st.session_state["lang"] == "ja" else "secondary"):
            st.session_state["lang"] = "ja"
            st.rerun()
    with lang_col2:
        if st.button("中文", use_container_width=True,
                      type="primary" if st.session_state["lang"] == "zh" else "secondary"):
            st.session_state["lang"] = "zh"
            st.rerun()

with top_left:
    st.title(t("title"))
    st.caption(t("caption"))

# -------------------------------------------------
# サイドバー（設定項目）
# -------------------------------------------------
with st.sidebar:
    st.header(t("settings_header"))

    uploaded_file = st.file_uploader(
        t("uploader_label"), type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input(t("width_label"), min_value=1, max_value=300, value=50, step=1)
    with col2:
        height = st.number_input(t("height_label"), min_value=1, max_value=300, value=50, step=1)

    use_max_colors = st.checkbox(t("use_max_colors_label"), value=True)
    max_colors = None
    if use_max_colors:
        max_colors = st.number_input(
            t("max_colors_label"), min_value=1, max_value=1024, value=20, step=1
        )

    run_button = st.button(t("run_button"), type="primary", use_container_width=True)

# -------------------------------------------------
# アップロード画像プレビュー
# -------------------------------------------------
if uploaded_file is not None:
    st.subheader(t("preview_subheader"))
    preview_img = Image.open(uploaded_file).convert("RGB")
    st.image(preview_img, width=250)
    # ファイルポインタを先頭に戻しておく（後でrun_agentに再度読み込ませるため）
    uploaded_file.seek(0)

# -------------------------------------------------
# 実行処理
# -------------------------------------------------
if run_button:
    if uploaded_file is None:
        st.error(t("error_no_upload"))
    else:
        with st.spinner(t("spinner_text")):
            image_bytes = uploaded_file.read()
            result = run_agent(
                image_bytes=image_bytes,
                target_width=int(width),
                target_height=int(height),
                max_colors=int(max_colors) if max_colors else None,
                color_db_path=COLOR_DB_PATH,
            )

        if result.get("error"):
            st.error(t("error_prefix") + str(result["error"]))
        else:
            # 結果をセッションに保存（再実行しても消えないように）
            st.session_state["last_result"] = result

# -------------------------------------------------
# 結果表示
# -------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.success(t("success_text"))

    col_img, col_list = st.columns([2, 1])

    with col_img:
        st.subheader(t("image_subheader"))
        # ビーズ図案・座標(A,B,C.../1,2,3...)・色番号一覧が
        # すべて1枚に統合されたPNG（agent_bead.py側で描画済み）
        output_image: Image.Image = result["output_image"]
        st.image(output_image, use_container_width=True)

        # 図面PNGダウンロード（統合済み画像そのまま。別ファイル化はしない）
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        st.download_button(
            t("download_png_button"),
            data=buf.getvalue(),
            file_name="bead_pattern.png",
            mime="image/png",
        )

    with col_list:
        st.subheader(t("legend_subheader"))
        color_counts = result["color_counts"]
        color_db = {c["code"]: c for c in result["color_db"]}

        table_rows = []
        total = 0
        for code, count in color_counts.items():
            name = color_db.get(code, {}).get("name", "")
            table_rows.append({
                t("col_code"): code,
                t("col_name"): name,
                t("col_count"): count,
            })
            total += count

        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        st.markdown(f"**{t('total_text').format(total=total, colors=len(color_counts))}**")

        # テキスト出力ダウンロード（既に実装済み）
        st.download_button(
            t("download_text_button"),
            data=result["result_text"],
            file_name="bead_list.txt",
            mime="text/plain",
        )

else:
    st.info(t("info_initial"))


# =================================================
# --- 拡張実装のヒント（コメントのみ・現状は未使用） ---
# =================================================
# 1マス手動修正機能を追加する場合の例:
#
#   import numpy as np
#   from agent_bead import render_grid_node, count_colors_node, build_result_text_node
#
#   idx_map = result["color_map"]  # (H, W) の numpy配列
#   x, y = 3, 5                    # 修正したいセル座標
#   new_color_code = "S07"
#   code_to_idx = {c["code"]: i for i, c in enumerate(result["color_db"])}
#   idx_map[y, x] = code_to_idx[new_color_code]
#   result["color_map"] = idx_map
#   result = count_colors_node(result)
#   result = render_grid_node(result)
#   result = build_result_text_node(result)
#   st.session_state["last_result"] = result
#   st.rerun()
#
# 言語を追加する場合の例（例: 英語を追加）:
#   TRANSLATIONS["en"] = { "page_title": "Iron Bead Converter", ... }
#   上部の言語切り替えボタン部分に "English" ボタンを追加するだけでよい。
