#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_scan_xml_structure.py

作用：
1. 扫描 CIM/RDF XML 文件中的所有标签；
2. 统计每种标签出现次数；
3. 统计每种父标签下面有哪些直接子标签；
4. 输出：
   - tag_summary.csv
   - tag_children_summary.csv

用法：
    python 01_scan_xml_structure.py your_file.xml -o output

只统计 cim 标签：
    python 01_scan_xml_structure.py your_file.xml -o output --prefixes cim

统计 cim 和 cimEx 标签：
    python 01_scan_xml_structure.py your_file.xml -o output --prefixes cim,cimEx
"""

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


# 常见命名空间预置
DEFAULT_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "cim": "http://iec.ch/TC57/2003/CIM-schema-cim10#",
    "cimEx": "http://dongfang-china.com/UAI#",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Scan CIM/RDF XML structure.")
    parser.add_argument("xml_file", nargs="?",
        default="data/raw/XML/zs_dicp_model-20260201.xml", help="输入 XML 文件路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="输出目录，默认 output"
    )
    parser.add_argument(
        "--prefixes",
        default="",
        help=(
            "只统计指定前缀的标签，多个用逗号分隔。"
            "例如：--prefixes cim 或 --prefixes cim,cimEx。"
            "默认不限制，统计所有标签。"
        )
    )
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_qname(tag):
    """
    ElementTree 中标签通常长这样：
        {namespace_uri}local_name

    例如：
        {http://iec.ch/TC57/2003/CIM-schema-cim10#}Analog

    返回：
        namespace_uri, local_name
    """
    if tag.startswith("{") and "}" in tag:
        uri, local = tag[1:].split("}", 1)
        return uri, local
    return "", tag


def build_short_tag(tag, uri_to_prefix):
    """
    把 ElementTree 的完整标签转换成更直观的形式。

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


def get_prefix(tag, uri_to_prefix):
    uri, _ = split_qname(tag)
    return uri_to_prefix.get(uri, "") if uri else ""


def should_keep_tag(tag, uri_to_prefix, allowed_prefixes):
    """
    根据 --prefixes 参数判断是否保留该标签。
    """
    if not allowed_prefixes:
        return True

    prefix = get_prefix(tag, uri_to_prefix)
    return prefix in allowed_prefixes


def write_tag_summary(output_file, tag_counter, uri_to_prefix):
    rows = []

    for raw_tag, count in tag_counter.most_common():
        uri, local = split_qname(raw_tag)
        prefix = uri_to_prefix.get(uri, "") if uri else ""
        short_tag = build_short_tag(raw_tag, uri_to_prefix)

        rows.append({
            "tag": short_tag,
            "count": count,
            "prefix": prefix,
            "local_name": local,
            "namespace_uri": uri,
            "raw_tag": raw_tag,
        })

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "count",
                "prefix",
                "local_name",
                "namespace_uri",
                "raw_tag",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_children_summary(output_file, children_counter, uri_to_prefix):
    rows = []

    for parent_raw_tag, child_counter in children_counter.items():
        parent_uri, parent_local = split_qname(parent_raw_tag)
        parent_prefix = uri_to_prefix.get(parent_uri, "") if parent_uri else ""
        parent_short_tag = build_short_tag(parent_raw_tag, uri_to_prefix)

        for child_raw_tag, count in child_counter.most_common():
            child_uri, child_local = split_qname(child_raw_tag)
            child_prefix = uri_to_prefix.get(child_uri, "") if child_uri else ""
            child_short_tag = build_short_tag(child_raw_tag, uri_to_prefix)

            rows.append({
                "parent_tag": parent_short_tag,
                "child_tag": child_short_tag,
                "count": count,

                "parent_prefix": parent_prefix,
                "parent_local_name": parent_local,
                "parent_namespace_uri": parent_uri,

                "child_prefix": child_prefix,
                "child_local_name": child_local,
                "child_namespace_uri": child_uri,

                "parent_raw_tag": parent_raw_tag,
                "child_raw_tag": child_raw_tag,
            })

    # 按 parent_tag、count 排序，方便看
    rows.sort(key=lambda x: (x["parent_tag"], -x["count"], x["child_tag"]))

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parent_tag",
                "child_tag",
                "count",

                "parent_prefix",
                "parent_local_name",
                "parent_namespace_uri",

                "child_prefix",
                "child_local_name",
                "child_namespace_uri",

                "parent_raw_tag",
                "child_raw_tag",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def scan_xml(xml_file, output_dir, prefixes):
    ensure_dir(output_dir)

    # 命名空间映射
    prefix_to_uri = dict(DEFAULT_NS)
    uri_to_prefix = {uri: prefix for prefix, uri in prefix_to_uri.items()}

    allowed_prefixes = set()
    if prefixes:
        allowed_prefixes = set([p.strip() for p in prefixes.split(",") if p.strip()])

    tag_counter = Counter()
    children_counter = defaultdict(Counter)

    # 用栈记录当前父标签
    tag_stack = []

    # iterparse 支持 start-ns 事件，用来读取 XML 文件里真实声明的命名空间
    context = ET.iterparse(xml_file, events=("start-ns", "start", "end"))

    for event, elem in context:
        if event == "start-ns":
            prefix, uri = elem
            if prefix is None:
                prefix = ""
            if prefix == "":
                prefix = "default"

            # 如果文件中声明了命名空间，以文件为准
            prefix_to_uri[prefix] = uri
            uri_to_prefix[uri] = prefix

        elif event == "start":
            current_tag = elem.tag

            # 统计当前标签
            if should_keep_tag(current_tag, uri_to_prefix, allowed_prefixes):
                tag_counter[current_tag] += 1

            # 统计父子关系
            if tag_stack:
                parent_tag = tag_stack[-1]

                parent_keep = should_keep_tag(parent_tag, uri_to_prefix, allowed_prefixes)
                child_keep = should_keep_tag(current_tag, uri_to_prefix, allowed_prefixes)

                # 如果限制了 prefixes，那么父子都在范围内才统计
                # 如果不限制，则全部统计
                if not allowed_prefixes:
                    children_counter[parent_tag][current_tag] += 1
                else:
                    if parent_keep and child_keep:
                        children_counter[parent_tag][current_tag] += 1

            tag_stack.append(current_tag)

        elif event == "end":
            if tag_stack:
                tag_stack.pop()

            # 清理元素，降低内存占用
            elem.clear()

    tag_summary_file = os.path.join(output_dir, "tag_summary.csv")
    children_summary_file = os.path.join(output_dir, "tag_children_summary.csv")

    write_tag_summary(tag_summary_file, tag_counter, uri_to_prefix)
    write_children_summary(children_summary_file, children_counter, uri_to_prefix)

    print("扫描完成。")
    print(f"标签统计输出：{tag_summary_file}")
    print(f"父子标签统计输出：{children_summary_file}")
    print(f"标签种类数量：{len(tag_counter)}")


def main():
    args = parse_args()
    scan_xml(args.xml_file, args.output_dir, args.prefixes)


if __name__ == "__main__":
    main()