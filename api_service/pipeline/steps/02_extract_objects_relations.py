#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_extract_objects_relations.py

作用：
1. 提取 XML 中所有 rdf:ID / rdf:about 对象；
2. 提取对象名称 cim:IdentifiedObject.name；
3. 提取对象 mRID；
4. 提取所有 rdf:resource 引用关系；
5. 输出：
   - objects.csv
   - relations.csv

用法：
    python 02_extract_objects_relations.py your_file.xml -o output
"""

import argparse
import csv
import os
import xml.etree.ElementTree as ET


DEFAULT_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "cim": "http://iec.ch/TC57/2003/CIM-schema-cim10#",
    "cimEx": "http://dongfang-china.com/UAI#",
}

RDF_URI = DEFAULT_NS["rdf"]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CIM/RDF objects and relations.")
    parser.add_argument("xml_file", nargs="?",
        default="data/raw/XML/zs_dicp_model-20260201.xml", help="输入 XML 文件路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="输出目录，默认 output"
    )
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_qname(tag):
    """
    把 ElementTree 的标签拆成 namespace_uri 和 local_name。

    输入：
        {http://iec.ch/TC57/2003/CIM-schema-cim10#}Analog

    输出：
        namespace_uri = http://iec.ch/TC57/2003/CIM-schema-cim10#
        local_name = Analog
    """
    if tag.startswith("{") and "}" in tag:
        uri, local = tag[1:].split("}", 1)
        return uri, local
    return "", tag


def build_short_tag(tag, uri_to_prefix):
    """
    把完整标签转换成短标签。

    输入：
        {http://iec.ch/TC57/2003/CIM-schema-cim10#}Analog

    输出：
        cim:Analog
    """
    uri, local = split_qname(tag)
    if uri:
        prefix = uri_to_prefix.get(uri, "")
        if prefix:
            return f"{prefix}:{local}"
        else:
            return local
    return local


def get_attr(elem, namespace_uri, attr_name):
    """
    获取带命名空间的属性。

    例如：
        rdf:ID
        rdf:about
        rdf:resource

    在 ElementTree 里实际是：
        {rdf_uri}ID
        {rdf_uri}about
        {rdf_uri}resource
    """
    return elem.attrib.get(f"{{{namespace_uri}}}{attr_name}")


def normalize_ref(value):
    """
    规范化 rdf:ID / rdf:about / rdf:resource。

    常见输入：
        ME-66970
        #ME-66970
        http://xxx#ME-66970

    输出：
        ME-66970
    """
    if value is None:
        return ""

    value = value.strip()

    if value.startswith("#"):
        return value[1:]

    if "#" in value:
        return value.split("#")[-1]

    return value


def write_objects(output_file, objects):
    fieldnames = [
        "object_id",
        "object_type",
        "name",
        "mrid",
        "description",
        "id_source",
        "namespace_uri",
        "raw_tag",
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(objects)


def write_relations(output_file, relations):
    fieldnames = [
        "source_id",
        "source_type",
        "relation",
        "target_id",
        "target_raw",
        "relation_namespace_uri",
        "relation_raw_tag",
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(relations)


def extract_objects_relations(xml_file, output_dir):
    ensure_dir(output_dir)

    prefix_to_uri = dict(DEFAULT_NS)
    uri_to_prefix = {uri: prefix for prefix, uri in prefix_to_uri.items()}

    objects = []
    relations = []

    # 当前正在解析的对象栈
    # 每个对象上下文结构：
    # {
    #   "elem": elem,
    #   "object_id": "...",
    #   "object_type": "...",
    #   "name": "",
    #   "mrid": "",
    #   "description": "",
    #   ...
    # }
    object_stack = []

    context = ET.iterparse(xml_file, events=("start-ns", "start", "end"))

    for event, elem in context:
        if event == "start-ns":
            prefix, uri = elem
            if prefix is None:
                prefix = ""
            if prefix == "":
                prefix = "default"

            prefix_to_uri[prefix] = uri
            uri_to_prefix[uri] = prefix

        elif event == "start":
            raw_tag = elem.tag
            namespace_uri, local_name = split_qname(raw_tag)
            short_tag = build_short_tag(raw_tag, uri_to_prefix)

            rdf_id = get_attr(elem, RDF_URI, "ID")
            rdf_about = get_attr(elem, RDF_URI, "about")
            rdf_resource = get_attr(elem, RDF_URI, "resource")

            # 1. 如果当前标签有 rdf:resource，说明它是一个引用关系
            #    例如：
            #    <cim:Measurement.PowerSystemResource rdf:resource="#LSZ.FH708"/>
            #
            #    当前对象栈顶部就是 source，resource 指向 target
            if rdf_resource and object_stack:
                source_ctx = object_stack[-1]

                relations.append({
                    "source_id": source_ctx["object_id"],
                    "source_type": source_ctx["object_type"],
                    "relation": short_tag,
                    "target_id": normalize_ref(rdf_resource),
                    "target_raw": rdf_resource,
                    "relation_namespace_uri": namespace_uri,
                    "relation_raw_tag": raw_tag,
                })

            # 2. 如果当前标签有 rdf:ID 或 rdf:about，说明它是一个对象
            #
            #    例如：
            #    <cim:Analog rdf:ID="ME-66970">
            #    <cim:Breaker rdf:about="#BRK001">
            if rdf_id or rdf_about:
                if rdf_id:
                    object_id = normalize_ref(rdf_id)
                    id_source = "rdf:ID"
                else:
                    object_id = normalize_ref(rdf_about)
                    id_source = "rdf:about"

                ctx = {
                    "elem": elem,
                    "object_id": object_id,
                    "object_type": short_tag,
                    "name": "",
                    "mrid": "",
                    "description": "",
                    "id_source": id_source,
                    "namespace_uri": namespace_uri,
                    "raw_tag": raw_tag,
                }

                object_stack.append(ctx)

        elif event == "end":
            raw_tag = elem.tag
            namespace_uri, local_name = split_qname(raw_tag)

            text = elem.text.strip() if elem.text else ""

            # 3. 如果当前结束的是普通属性标签，尝试把文本写入当前对象
            #
            # 常见：
            # <cim:IdentifiedObject.name>708龙田线</cim:IdentifiedObject.name>
            # <cim:IdentifiedObject.mRID>xxx</cim:IdentifiedObject.mRID>
            # <cim:IdentifiedObject.description>xxx</cim:IdentifiedObject.description>
            if object_stack and text:
                current_ctx = object_stack[-1]

                # 注意：不能把对象自身标签的 text 当作 name
                if current_ctx["elem"] is not elem:
                    if local_name == "IdentifiedObject.name" or local_name.endswith(".name") or local_name == "name":
                        # 优先使用 IdentifiedObject.name
                        if not current_ctx["name"] or local_name == "IdentifiedObject.name":
                            current_ctx["name"] = text

                    elif local_name == "IdentifiedObject.mRID" or local_name.endswith(".mRID") or local_name == "mRID":
                        if not current_ctx["mrid"]:
                            current_ctx["mrid"] = text

                    elif (
                        local_name == "IdentifiedObject.description"
                        or local_name.endswith(".description")
                        or local_name == "description"
                    ):
                        if not current_ctx["description"]:
                            current_ctx["description"] = text

            # 4. 如果当前结束的是对象本身，则弹出对象栈并保存
            if object_stack and object_stack[-1]["elem"] is elem:
                ctx = object_stack.pop()

                objects.append({
                    "object_id": ctx["object_id"],
                    "object_type": ctx["object_type"],
                    "name": ctx["name"],
                    "mrid": ctx["mrid"],
                    "description": ctx["description"],
                    "id_source": ctx["id_source"],
                    "namespace_uri": ctx["namespace_uri"],
                    "raw_tag": ctx["raw_tag"],
                })

            # 清理元素，减少内存占用
            elem.clear()

    objects_file = os.path.join(output_dir, "objects.csv")
    relations_file = os.path.join(output_dir, "relations.csv")

    write_objects(objects_file, objects)
    write_relations(relations_file, relations)

    print("提取完成。")
    print(f"对象表输出：{objects_file}")
    print(f"关系表输出：{relations_file}")
    print(f"对象数量：{len(objects)}")
    print(f"关系数量：{len(relations)}")


def main():
    args = parse_args()
    extract_objects_relations(args.xml_file, args.output_dir)


if __name__ == "__main__":
    main()