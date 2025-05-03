import gradio as gr
from gradio_app import OCRProcessorNo, OCRProcessorSelect, OCRProcessorProgram
import tempfile
import os

# 初始化处理器（在全局范围只初始化一次）
processor_no = OCRProcessorNo()
processor_select = OCRProcessorSelect()
processor_program = OCRProcessorProgram()


def process_学号(image):
    try:
        # 保存上传的图片到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            results = processor_no.process_image(tmp_file.name)
        os.unlink(tmp_file.name)
        return "\n".join(results) if results else "未识别到学号"
    except Exception as e:
        return f"处理失败：{str(e)}"


def process_选择题(image):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            results = processor_select.process_image(tmp_file.name)
        os.unlink(tmp_file.name)
        return "\n".join(results) if results else "未识别到选择题答案"
    except Exception as e:
        return f"处理失败：{str(e)}"


def process_程序题(image):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            results = processor_program.process_image(tmp_file.name)
        os.unlink(tmp_file.name)
        return "\n".join(results) if results else "未识别到程序题答案"
    except Exception as e:
        return f"处理失败：{str(e)}"


with gr.Blocks(title="答题卡识别系统") as demo:
    gr.Markdown("# 答题卡识别系统")

    with gr.Tabs():
        with gr.TabItem("学号识别"):
            with gr.Row():
                with gr.Column():
                    upload_no = gr.Image(label="上传学号区域图片", type="pil")
                    btn_no = gr.Button("开始识别")
                result_no = gr.Textbox(label="识别结果", lines=5)
            btn_no.click(fn=process_学号, inputs=upload_no, outputs=result_no)

        with gr.TabItem("选择题识别"):
            with gr.Row():
                with gr.Column():
                    upload_select = gr.Image(label="上传选择题区域图片", type="pil")
                    btn_select = gr.Button("开始识别")
                result_select = gr.Textbox(label="识别结果", lines=10)
            btn_select.click(fn=process_选择题, inputs=upload_select, outputs=result_select)

        with gr.TabItem("程序题识别"):
            with gr.Row():
                with gr.Column():
                    upload_program = gr.Image(label="上传程序题区域图片", type="pil")
                    btn_program = gr.Button("开始识别")
                result_program = gr.Textbox(label="识别结果", lines=15)
            btn_program.click(fn=process_程序题, inputs=upload_program, outputs=result_program)

if __name__ == "__main__":
    demo.launch()