"""
app.py
=================================================
アイロンビーズ変換ツール - Streamlit フロントエンド

agent_bead.py の run_agent() を呼び出して、
画像アップロード → 図面生成 → 結果表示 までを行う。

拡張ポイント:
  - 「1マス手動修正」機能を足す場合は、st.session_state に
    color_map / color_db を保持し、セル選択UI(例: st.columns + selectbox)
    から color_map を書き換え、agent_bead.render_grid_node() /
    agent_bead.count_colors_node() / build_result_text_node() を
    直接呼び直して再描画すればよい（このファイルの一番下にヒントを記載）。
  - 「Hama色に切り替え」機能を足す場合は、color_db_path を
    hama_color.json 等に差し替えるセレクトボックスを追加すればよい。
  - 「テキスト出力ダウンロード」は既に実装済み（st.download_button）。
=================================================
"""

import io
import os

import streamlit as st
from PIL import Image

from agent_bead import run_agent

# -------------------------------------------------
# 基本設定
# -------------------------------------------------
st.set_page_config(page_title="アイロンビーズ変換ツール", page_icon="🧵", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLOR_DB_PATH = os.path.join(BASE_DIR, "artkal_color.json")

st.title("🧵 アイロンビーズ変換ツール")
st.caption("画像をアップロードすると、Artkalビーズの色番号に変換した図案を作成します。")

# -------------------------------------------------
# サイドバー（設定項目）
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ 設定")

    uploaded_file = st.file_uploader(
        "画像をアップロード", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("横幅（マス数）", min_value=1, max_value=300, value=50, step=1)
    with col2:
        height = st.number_input("縦幅（マス数）", min_value=1, max_value=300, value=50, step=1)

    use_max_colors = st.checkbox("使用色数を制限する", value=True)
    max_colors = None
    if use_max_colors:
        max_colors = st.number_input(
            "最大使用色数", min_value=1, max_value=35, value=20, step=1
        )

    run_button = st.button("🚀 変換実行", type="primary", use_container_width=True)

# -------------------------------------------------
# アップロード画像プレビュー
# -------------------------------------------------
if uploaded_file is not None:
    st.subheader("アップロード画像プレビュー")
    preview_img = Image.open(uploaded_file).convert("RGB")
    st.image(preview_img, width=250)
    # ファイルポインタを先頭に戻しておく（後でrun_agentに再度読み込ませるため）
    uploaded_file.seek(0)

# -------------------------------------------------
# 実行処理
# -------------------------------------------------
if run_button:
    if uploaded_file is None:
        st.error("先に画像をアップロードしてください。")
    else:
        with st.spinner("エージェントが変換中です..."):
            image_bytes = uploaded_file.read()
            result = run_agent(
                image_bytes=image_bytes,
                target_width=int(width),
                target_height=int(height),
                max_colors=int(max_colors) if max_colors else None,
                color_db_path=COLOR_DB_PATH,
            )

        if result.get("error"):
            st.error(f"エラーが発生しました: {result['error']}")
        else:
            # 結果をセッションに保存（再実行しても消えないように）
            st.session_state["last_result"] = result

# -------------------------------------------------
# 結果表示
# -------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.success("変換が完了しました！")

    col_img, col_list = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ アイロンビーズ図面")
        output_image: Image.Image = result["output_image"]
        st.image(output_image, use_container_width=True)

        # 図面PNGダウンロード
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        st.download_button(
            "図面PNGをダウンロード",
            data=buf.getvalue(),
            file_name="bead_pattern.png",
            mime="image/png",
        )

    with col_list:
        st.subheader("🎨 使用ビーズ一覧")
        color_counts = result["color_counts"]
        color_db = {c["code"]: c for c in result["color_db"]}

        table_rows = []
        total = 0
        for code, count in color_counts.items():
            name = color_db.get(code, {}).get("name", "")
            table_rows.append({"色番号": code, "色名": name, "必要個数": count})
            total += count

        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        st.markdown(f"**合計ビーズ数: {total} 個 / 使用色数: {len(color_counts)} 色**")

        # テキスト出力ダウンロード（拡張要件：既に実装済み）
        st.download_button(
            "色番号・個数リストをテキストでダウンロード",
            data=result["result_text"],
            file_name="bead_list.txt",
            mime="text/plain",
        )

else:
    st.info("左側のサイドバーから画像をアップロードし、「変換実行」を押してください。")


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
