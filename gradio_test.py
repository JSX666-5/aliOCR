import gradio as gr

# 实例化三个处理器
processor_no = OCRProcessorNo()
processor_select = OCRProcessorSelect()
processor_program = OCRProcessorProgram()


def process_no(image):
    temp_path = "temp_no.jpg"
    cv2.imwrite(temp_path, image)
    results = processor_no.process_image(temp_path)
    return "\n".join(results) if results else "未识别到学号"


def process_select(image):
    temp_path = "temp_select.jpg"
    cv2.imwrite(temp_path, image)
    results = processor_select.process_image(temp_path)
    return "\n".join(results) if results else "未识别到选择题答案"


def process_program(image):
    temp_path = "temp_program.jpg"
    cv2.imwrite(temp_path, image)
    results = processor_program.process_image(temp_path)
    return "\n".join(results) if results else "未识别到编程题答案"


with gr.Blocks(title="三合一智能批改系统") as demo:
    gr.Markdown("## 📚 试卷批改系统 - 独立模块版")

    with gr.Tabs():
        # 学号识别模块
        with gr.TabItem("🏷️ 学号识别"):
            with gr.Row():
                with gr.Column():
                    input_no = gr.Image(label="上传包含学号的区域", type="numpy")
                    btn_no = gr.Button("识别学号", variant="primary")
                output_no = gr.Textbox(label="学号识别结果", lines=3)

        # 选择题识别模块
        with gr.TabItem("🔘 选择题识别"):
            with gr.Row():
                with gr.Column():
                    input_select = gr.Image(label="上传选择题答题区域", type="numpy")
                    btn_select = gr.Button("识别选择题", variant="primary")
                output_select = gr.Textbox(label="选择题识别结果", lines=5)

        # 编程题识别模块
        with gr.TabItem("💻 编程题识别"):
            with gr.Row():
                with gr.Column():
                    input_program = gr.Image(label="上传编程题答题区域", type="numpy")
                    btn_program = gr.Button("识别编程题", variant="primary")
                output_program = gr.Textbox(label="编程题识别结果", lines=10)

    # 绑定事件
    btn_no.click(fn=process_no, inputs=input_no, outputs=output_no)
    btn_select.click(fn=process_select, inputs=input_select, outputs=output_select)
    btn_program.click(fn=process_program, inputs=input_program, outputs=output_program)

if __name__ == "__main__":
    demo.launch()