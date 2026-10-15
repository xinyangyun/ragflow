#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  本文件只能按照许可证规定使用。
#  你可以在以下地址获取许可证副本：
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  除非适用法律要求或书面同意，否则本文件按“原样”提供，
#  不附带任何明示或默示的保证或条件。
#

# 导入日志模块，用于记录解析过程和异常信息。
import logging
# 导入正则表达式模块，用于识别文件扩展名和清理文本。
import re
# 导入内存字节流，用于把二进制 DOC 文件交给 Tika。
from io import BytesIO

# 导入通用文本读取函数。
from deepdoc.parser.utils import get_text
# 导入通用解析器模块，用于复用其 DOCX 解析器。
from rag.app import naive
# 导入 PDF 解析器映射和纯文本兜底解析函数。
from rag.app.naive import by_plaintext, PARSERS
# 导入 PDF 最大页码常量。
from common.constants import MAXIMUM_PAGE_NUMBER
# 导入布局识别器配置规范化函数。
from common.parser_config_utils import normalize_layout_recognizer
# 导入书籍切块需要的 NLP 和媒体处理函数。
from rag.nlp import bullets_category, is_english, remove_contents_table, hierarchical_merge, make_colon_as_title, naive_merge, random_choices, tokenize_table, tokenize_chunks, attach_media_context
# 导入 RAGFlow 分词器。
from rag.nlp import rag_tokenizer
# 导入 PDF 和 HTML 解析器基类。
from deepdoc.parser import PdfParser, HtmlParser
# 导入 DOCX 图片视觉增强包装器。
from deepdoc.parser.figure_parser import vision_figure_parser_docx_wrapper
# 导入 PIL 图片类型，用于过滤原始图片对象。
from PIL import Image
# 导入惰性图片类型，用于过滤延迟加载的图片对象。
from rag.utils.lazy_image import LazyImage


# 定义书籍专用 PDF 解析器。
class Pdf(PdfParser):
    # 定义 PDF 解析调用入口。
    def __call__(self, filename, binary=None, from_page=0, to_page=MAXIMUM_PAGE_NUMBER, zoomin=3, callback=None):
        # 延迟导入计时器，避免模块加载时引入额外开销。
        from timeit import default_timer as timer

        # 记录 OCR 阶段开始时间。
        start = timer()
        # 通知调用方开始 OCR。
        callback(msg="OCR started")
        # 把 PDF 页面渲染成图片并限定解析页范围。
        self.__images__(filename if not binary else binary, zoomin, from_page, to_page, callback)
        # 通知调用方 OCR 已完成并报告耗时。
        callback(msg="OCR finished ({:.2f}s)".format(timer() - start))

        # 记录版面分析阶段开始时间。
        start = timer()
        # 执行文本、表格、图片等版面识别。
        self._layouts_rec(zoomin)
        # 上报版面分析进度和耗时。
        callback(0.67, "Layout analysis ({:.2f}s)".format(timer() - start))
        # 输出版面分析的调试日志。
        logging.debug("layouts: {}".format(timer() - start))

        # 记录表格分析阶段开始时间。
        start = timer()
        # 执行表格结构识别和转换。
        self._table_transformer_job(zoomin)
        # 上报表格分析进度和耗时。
        callback(0.68, "Table analysis ({:.2f}s)".format(timer() - start))

        # 记录文本提取阶段开始时间。
        start = timer()
        # 合并同一文本区域中的文本框。
        self._text_merge()
        # 提取表格和图片，并保留 HTML 和位置信息。
        tbls = self._extract_table_figure(True, zoomin, True, True)
        # 对相邻的普通文本执行纵向合并。
        self._naive_vertical_merge()
        # 删除不在目标页范围内的文本框。
        self._filter_forpages()
        # 合并具有相同项目符号的相邻文本框。
        self._merge_with_same_bullet()
        # 上报文本提取进度和耗时。
        callback(0.8, "Text extraction ({:.2f}s)".format(timer() - start))

        # 返回带页面位置标签的文本段，以及表格/图片集合。
        return [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes], tbls


# 定义书籍文件的统一解析和切块入口。
def chunk(filename, binary=None, from_page=0, to_page=MAXIMUM_PAGE_NUMBER, lang="Chinese", callback=None, **kwargs):
    # 说明支持的文件类型以及长书籍建议分段解析的原因。
    """
    Supported file formats are docx, pdf, txt.
    Since a book is long and not all the parts are useful, if it's a PDF,
    please set up the page ranges for every book in order eliminate negative effects and save elapsed computing time.
    """
    # 从可选参数中读取解析配置，缺省时使用书籍解析默认值。
    parser_config = kwargs.get("parser_config", {"chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"})
    # 创建文档级元数据，并用去掉扩展名的文件名生成标题词元。
    doc = {"docnm_kwd": filename, "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))}
    # 为标题词元生成更细粒度的检索词元。
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    # 保存 PDF 解析器实例，后续文本切块可能需要它裁剪页面图片。
    pdf_parser = None
    # 初始化正文段和表格/图片列表。
    sections, tbls = [], []
    # 判断当前文件是否为 DOCX。
    if re.search(r"\.docx$", filename, re.IGNORECASE):
        # 通知调用方开始解析 DOCX。
        callback(0.1, "Start to parse.")
        # 创建通用 DOCX 解析器。
        doc_parser = naive.Docx()
        # TODO：目录表目前仍由后续目录清理函数处理。
        # 解析 DOCX 正文、图片和表格。
        main_sections = doc_parser(filename, binary=binary, from_page=from_page, to_page=to_page)

        # 为 DOCX 正文和表格准备独立列表。
        sections = []
        # 清空表格列表，避免复用旧数据。
        tbls = []
        # 遍历解析器返回的文本、图片和 HTML 表格。
        for text, image, html in main_sections:
            # 保存正文文本及其关联图片对象。
            sections.append((text, image))
            # 把 HTML 表格包装成统一的表格项格式。
            tbls.append(((None, html), ""))

        # 根据语言抽样结果删除 DOCX 中的目录内容。
        remove_contents_table(sections, eng=is_english(random_choices([t for t, _ in sections], k=200)))

        # 尝试使用视觉模型增强 DOCX 图片描述。
        tbls = vision_figure_parser_docx_wrapper(sections=sections, tbls=tbls, callback=callback, **kwargs)
        # 过滤掉无法直接参与文本切块的 PIL 图片和惰性图片对象。
        sections = [(item[0], item[1] if item[1] is not None else "") for item in sections if not isinstance(item[1], (Image.Image, LazyImage))]
        # 通知调用方 DOCX 解析完成。
        callback(0.8, "Finish parsing.")

    # 判断当前文件是否为 PDF。
    elif re.search(r"\.pdf$", filename, re.IGNORECASE):
        # 规范化布局识别器名称，并提取可能绑定的模型名。
        layout_recognizer, parser_model_name = normalize_layout_recognizer(parser_config.get("layout_recognize", "DeepDOC"))

        # 兼容旧配置中的布尔型布局识别开关。
        if isinstance(layout_recognizer, bool):
            # True 表示使用 DeepDOC，False 表示使用纯文本解析。
            layout_recognizer = "DeepDOC" if layout_recognizer else "Plain Text"

        # 统一转成小写后用于查找解析器。
        name = layout_recognizer.strip().lower()
        # 根据名称选择解析器，找不到时退回纯文本解析。
        parser = PARSERS.get(name, by_plaintext)
        # 通知调用方开始解析 PDF。
        callback(0.1, "Start to parse.")

        # 调用选定的 PDF 解析器获得正文、表格和解析器实例。
        sections, tables, pdf_parser = parser(
            # 传入文件路径。
            filename=filename,
            # 传入文件二进制内容。
            binary=binary,
            # 传入起始页码。
            from_page=from_page,
            # 传入结束页码。
            to_page=to_page,
            # 传入文档语言。
            lang=lang,
            # 传入进度回调函数。
            callback=callback,
            # 指定书籍专用 PDF 类。
            pdf_cls=Pdf,
            # 传入规范化后的布局识别器。
            layout_recognizer=layout_recognizer,
            # 传入 MinerU 模型名称（如果有）。
            mineru_llm_name=parser_model_name,
            # 传入 PaddleOCR 模型名称（如果有）。
            paddleocr_llm_name=parser_model_name,
            # 透传其他解析参数。
            **kwargs,
        )

        # 如果正文和表格都为空，直接返回空结果。
        if not sections and not tables:
            return []

        # 对结构化解析器关闭二次 token 合并，尽量保持其原始分块。
        if name in ["tcadp", "docling", "mineru", "paddleocr"]:
            # 0 表示不再按 token 数强制合并。
            parser_config["chunk_token_num"] = 0

        # 通知调用方 PDF 解析阶段完成。
        callback(0.8, "Finish parsing.")

    # 判断当前文件是否为纯文本文件。
    elif re.search(r"\.txt$", filename, re.IGNORECASE):
        # 通知调用方开始解析 TXT。
        callback(0.1, "Start to parse.")
        # 读取文件内容并处理可能的二进制输入。
        txt = get_text(filename, binary)
        # 先按换行切分成文本行。
        sections = txt.split("\n")
        # 丢弃空行，并统一包装为正文段格式。
        sections = [(line, "") for line in sections if line]
        # 删除文本中的目录内容。
        remove_contents_table(sections, eng=is_english(random_choices([t for t, _ in sections], k=200)))
        # 通知调用方 TXT 解析完成。
        callback(0.8, "Finish parsing.")

    # 判断当前文件是否为 HTML 文件。
    elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
        # 通知调用方开始解析 HTML。
        callback(0.1, "Start to parse.")
        # 调用 HTML 解析器得到文本段。
        sections = HtmlParser()(filename, binary)
        # 丢弃空行，并统一包装为正文段格式。
        sections = [(line, "") for line in sections if line]
        # 删除 HTML 中可能存在的目录内容。
        remove_contents_table(sections, eng=is_english(random_choices([t for t, _ in sections], k=200)))
        # 通知调用方 HTML 解析完成。
        callback(0.8, "Finish parsing.")

    # 判断当前文件是否为旧版 DOC 文件。
    elif re.search(r"\.doc$", filename, re.IGNORECASE):
        # 通知调用方开始解析 DOC。
        callback(0.1, "Start to parse.")
        # 尝试导入 Apache Tika 解析器。
        try:
            # Tika 用于读取旧版二进制 DOC 文档。
            from tika import parser as tika_parser
        # 捕获 Tika 未安装或导入失败的异常。
        except Exception as e:
            # 上报不支持 DOC 解析的进度信息。
            callback(0.8, f"tika not available: {e}. Unsupported .doc parsing.")
            # 记录 DOC 解析不可用的警告。
            logging.warning(f"tika not available: {e}. Unsupported .doc parsing for {filename}.")
            # 返回空列表，跳过当前文件。
            return []

        # 将二进制内容包装成文件流供 Tika 读取。
        binary = BytesIO(binary)
        # 使用 Tika 从内存文件流中提取文档内容。
        doc_parsed = tika_parser.from_buffer(binary)
        # 只有提取到正文内容时才继续处理。
        if doc_parsed.get("content", None) is not None:
            # 按换行切分 Tika 返回的正文。
            sections = doc_parsed["content"].split("\n")
            # 丢弃空行，并统一包装为正文段格式。
            sections = [(line, "") for line in sections if line]
            # 删除 DOC 中可能存在的目录内容。
            remove_contents_table(sections, eng=is_english(random_choices([t for t, _ in sections], k=200)))
            # 通知调用方 DOC 解析完成。
            callback(0.8, "Finish parsing.")

    # 对不支持的扩展名抛出明确异常。
    else:
        # 当前书籍解析器不处理其他文件类型。
        raise NotImplementedError("file type not supported yet(doc, docx, pdf, txt supported)")

    # 把以冒号结尾的长文本前缀识别为标题。
    make_colon_as_title(sections)
    # 从随机抽样文本中判断是否存在章节编号或项目符号结构。
    bull = bullets_category([t for t in random_choices([t for t, _ in sections], k=100)])
    # 如果存在层级符号，则按层级树合并章节。
    if bull >= 0:
        # 将层级合并返回的文本列表拼接成完整字符串块。
        chunks = ["\n".join(ck) for ck in hierarchical_merge(bull, sections, 5)]
    # 如果不存在层级符号，则走普通 token 数切块。
    else:
        # 临时拆分正文中的位置标签，保留可能存在的标签信息。
        sections = [s.split("@") for s, _ in sections]
        # 将拆分结果重新包装为正文和位置后缀二元组。
        sections = [(pr[0], "@" + pr[1]) if len(pr) == 2 else (pr[0], "") for pr in sections]
        # 按 token 上限和分隔符合并成普通文本块。
        chunks = naive_merge(sections, parser_config.get("chunk_token_num", 256), parser_config.get("delimiter", "\n。；！？"))

    # 根据调用方指定的语言决定后续分词模式。
    eng = lang.lower() == "english"

    # 先把表格和图片转换成独立的媒体 chunk。
    res = tokenize_table(tbls, doc, eng, language=lang)
    # 再把正文文本块转换成普通 RAG chunk，并追加到结果中。
    res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser, language=lang))
    # 读取表格上下文窗口大小，负数按零处理。
    table_ctx = max(0, int(parser_config.get("table_context_size", 0) or 0))
    # 读取图片上下文窗口大小，负数按零处理。
    image_ctx = max(0, int(parser_config.get("image_context_size", 0) or 0))
    # 只有配置了上下文窗口时才执行媒体上下文关联。
    if table_ctx or image_ctx:
        # 给表格或图片 chunk 附加附近正文上下文。
        attach_media_context(res, table_ctx, image_ctx)

    # 返回最终的正文、表格和图片 RAG chunk 列表。
    return res


# 允许直接从命令行运行本文件进行简单测试。
if __name__ == "__main__":
    # 导入命令行参数模块。
    import sys

    # 定义不输出任何内容的测试回调。
    def dummy(prog=None, msg=""):
        # 该回调仅用于满足 chunk 的进度通知接口。
        pass

    # 解析命令行传入文件的第 1 到第 10 页。
    chunk(sys.argv[1], from_page=1, to_page=10, callback=dummy)
