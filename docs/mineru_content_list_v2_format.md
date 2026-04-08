# MinerU content_list_v2.json 格式规范

## 概述

`content_list_v2.json` 是 MinerU 3.0+ 引入的新结构化输出格式。与旧版 `content_list.json` 相比，具有更好的结构化和程序化处理能力。

**文件命名格式**: `{original_filename}_content_list_v2.json`

## 版本特性

- **引入版本**: MinerU 3.0
- **后端支持**: 所有后端（Pipeline、VLM、Office）统一输出
- **状态**: 正式版本（非开发版）
- **兼容性**: 与 `content_list.json` 同时输出，向后兼容

## 核心设计

1. **按页面分组**: 顶层为数组的数组，外层索引 = 页码
2. **统一结构**: 所有元素使用 `type + content` 模式
3. **语义完整**: 每个元素是一个独立的语义单元

## 文件结构

```json
[
  [ // 页面 0 的内容项
    { "type": "...", "content": {...}, "bbox": [...] },
    { "type": "...", "content": {...}, "bbox": [...] }
  ],
  [ // 页面 1 的内容项
    { "type": "...", "content": {...}, "bbox": [...] }
  ]
]
```

**说明**:
- 外层数组: 按页面分组，索引从 0 开始
- 内层数组: 该页面的所有内容项（按阅读顺序）

## 通用字段

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `type` | string | ✅ | 内容类型（见下文类型表） |
| `content` | dict | ✅ | 该类型对应的结构化内容 |
| `bbox` | list[int] | ❌ | 边界框 `[x0, y0, x1, y1]`，坐标范围 0-1000 |
| `anchor` | string | ❌ | 锚点（部分 DOCX 标题或索引项可能包含） |

## 坐标系统

- **格式**: `[x0, y0, x1, y1]` - 左上角和右下角坐标
- **范围**: 0-1000（归一化到千分比）
- **原点**: 页面左上角

## 支持的内容类型

### 1. 文本类型

#### `title` - 标题块

```json
{
  "type": "title",
  "content": {
    "title_content": [
      { "type": "text", "content": "1 Introduction" }
    ],
    "level": 1
  },
  "bbox": [83, 121, 917, 156]
}
```

**content 字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `title_content` | list | 标题内容数组，元素为 `{type, content}` |
| `level` | int | 标题层级（1, 2, 3...） |

#### `paragraph` - 段落块

```json
{
  "type": "paragraph",
  "content": {
    "paragraph_content": [
      { "type": "text", "content": "段落文本内容..." }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `paragraph_content` | list | 段落内容数组 |

### 2. 数学类型

#### `equation_interline` - 行间公式

```json
{
  "type": "equation_interline",
  "content": {
    "math_content": "E = mc^2",
    "math_type": "latex"
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `math_content` | string | 数学公式内容 |
| `math_type` | string | 公式类型（如 "latex", "text"） |

### 3. 视觉类型

#### `image` - 图片

```json
{
  "type": "image",
  "content": {
    "image_path": "images/xxx.jpg",
    "image_caption": ["图 1. 说明文字"],
    "image_footnote": ["脚注内容"]
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `image_path` | string | 图片文件路径 |
| `image_caption` | list[string] | 图片说明 |
| `image_footnote` | list[string] | 图片脚注 |

#### `table` - 表格

```json
{
  "type": "table",
  "content": {
    "table_path": "images/xxx.jpg",
    "table_caption": ["表 1. 说明文字"],
    "table_footnote": ["脚注内容"],
    "table_body": "| 列1 | 列2 |..."
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `table_path` | string | 表格图片路径 |
| `table_caption` | list[string] | 表格说明 |
| `table_footnote` | list[string] | 表格脚注 |
| `table_body` | string | Markdown 格式的表格内容 |

#### `chart` - 图表

```json
{
  "type": "chart",
  "content": {
    "chart_path": "images/xxx.jpg",
    "chart_caption": ["图表说明"]
  },
  "bbox": [x0, y0, x1, y1]
}
```

#### `seal` - 印章

```json
{
  "type": "seal",
  "content": {
    "seal_path": "images/xxx.jpg"
  },
  "bbox": [x0, y0, x1, y1]
}
```

### 4. 代码类型

#### `code` - 代码块

```json
{
  "type": "code",
  "content": {
    "code_content": "def hello():\n    print('world')",
    "code_caption": ["代码 1. 示例"],
    "code_footnote": ["脚注"],
    "code_language": "python"
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段**:
| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `code_content` | string | ✅ | 代码内容 |
| `code_caption` | list[string] | ❌ | 代码说明 |
| `code_footnote` | list[string] | ❌ | 代码脚注 |
| `code_language` | string | ❌ | 编程语言 |

#### `algorithm` - 算法块

```json
{
  "type": "algorithm",
  "content": {
    "algorithm_content": "1: function ...",
    "algorithm_caption": ["算法 1. 描述"],
    "algorithm_footnote": ["说明"]
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段**:
| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `algorithm_content` | string | ✅ | 算法内容 |
| `algorithm_caption` | list[string] | ❌ | 算法说明 |
| `algorithm_footnote` | list[string] | ❌ | 算法脚注 |

### 5. 列表类型

#### `list` - 列表

```json
{
  "type": "list",
  "content": {
    "list_items": [
      { "type": "text", "content": "第一项" },
      { "type": "text", "content": "第二项" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

#### `index` - 索引

```json
{
  "type": "index",
  "content": {
    "list_items": [
      { "type": "text", "content": "索引项 1" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

**content 字段** (list/index):
| 字段 | 类型 | 描述 |
|------|------|------|
| `list_items` | list | 列表项数组，每项为 `{type, content}` |

### 6. 页面辅助类型

#### `page_header` - 页眉

```json
{
  "type": "page_header",
  "content": {
    "page_header_content": [
      { "type": "text", "content": "页眉内容" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

#### `page_footer` - 页脚

```json
{
  "type": "page_footer",
  "content": {
    "page_footer_content": [
      { "type": "text", "content": "页脚内容" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

#### `page_number` - 页码

```json
{
  "type": "page_number",
  "content": {
    "page_number_content": [
      { "type": "text", "content": "1" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

#### `page_aside_text` - 侧边注

```json
{
  "type": "page_aside_text",
  "content": {
    "page_aside_text_content": [
      { "type": "text", "content": "侧边注内容" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

#### `page_footnote` - 页面脚注

```json
{
  "type": "page_footnote",
  "content": {
    "page_footnote_content": [
      { "type": "text", "content": "* Corresponding author" }
    ]
  },
  "bbox": [x0, y0, x1, y1]
}
```

## 类型汇总表

| Type | 类别 | Content 根字段 |
|------|------|----------------|
| `title` | 文本 | `title_content`, `level` |
| `paragraph` | 文本 | `paragraph_content` |
| `equation_interline` | 数学 | `math_content`, `math_type` |
| `image` | 视觉 | `image_path`, `image_caption`, `image_footnote` |
| `table` | 视觉 | `table_path`, `table_caption`, `table_footnote`, `table_body` |
| `chart` | 视觉 | `chart_path`, `chart_caption` |
| `seal` | 视觉 | `seal_path` |
| `code` | 代码 | `code_content`, `code_caption`, `code_footnote`, `code_language` |
| `algorithm` | 代码 | `algorithm_content`, `algorithm_caption`, `algorithm_footnote` |
| `list` | 列表 | `list_items` |
| `index` | 列表 | `list_items` |
| `page_header` | 辅助 | `page_header_content` |
| `page_footer` | 辅助 | `page_footer_content` |
| `page_number` | 辅助 | `page_number_content` |
| `page_aside_text` | 辅助 | `page_aside_text_content` |
| `page_footnote` | 辅助 | `page_footnote_content` |

## 完整示例

```json
[
  [
    {
      "type": "title",
      "content": {
        "title_content": [
          { "type": "text", "content": "1 Introduction" }
        ],
        "level": 1
      },
      "bbox": [83, 121, 917, 156]
    },
    {
      "type": "paragraph",
      "content": {
        "paragraph_content": [
          { "type": "text", "content": "This is a paragraph..." }
        ]
      },
      "bbox": [83, 170, 917, 250]
    },
    {
      "type": "equation_interline",
      "content": {
        "math_content": "E = mc^2",
        "math_type": "latex"
      },
      "bbox": [200, 270, 800, 320]
    },
    {
      "type": "page_footnote",
      "content": {
        "page_footnote_content": [
          { "type": "text", "content": "* Corresponding author" }
        ]
      },
      "bbox": [71, 815, 915, 841]
    }
  ]
]
```

## 与 content_list.json 对比

| 特性 | content_list.json (v1) | content_list_v2.json (v2) |
|------|------------------------|---------------------------|
| 结构 | 扁平化数组 | 按页面分组的嵌套数组 |
| 语义单元 | type + 分散字段 | 统一的 type + content |
| 页面边界 | 仅通过 page_idx 字段 | 外层数组天然分组 |
| 字段命名 | 不统一（image_caption, code_body 等） | 统一后缀模式 |
| 后端一致性 | Pipeline 与 VLM 有差异 | 所有后端统一 |

## 适用场景

**推荐使用 content_list_v2.json 的场景**:
- 需要按页面处理文档
- 需要精确的页面边界信息
- 需要统一的程序化处理接口
- 需要区分不同类型的语义块

**继续使用 content_list.json 的场景**:
- 需要扁平化的简单结构
- 只关心内容不关心页面边界
- 兼容旧版本代码

## 参考链接

- [MinerU 官方文档 - Output File Format](https://opendatalab.github.io/MinerU/reference/output_files/)
