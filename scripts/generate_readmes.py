#!/usr/bin/env python3
"""Fetch challenge details from LeetGPU API and generate README.md files."""

import json
import re
import os
import urllib.request
from html.parser import HTMLParser

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LeetGPU")
API_BASE = "https://api.leetgpu.com/api/v1"

# Mapping: folder_name -> (url_slug, difficulty_display)
# The url_slug is used for the LeetGPU link
FOLDER_TO_SLUG = {
    # Easy (skip 01-04, they already have READMEs)
    "Easy/05_Matrix_Addition": ("matrix-addition", "Easy"),
    "Easy/06_1D_Convolution": ("1d-convolution", "Easy"),
    "Easy/07_Reverse_Array": ("reverse-array", "Easy"),
    "Easy/08_ReLU": ("relu", "Easy"),
    "Easy/09_Leaky_ReLU": ("leaky-relu", "Easy"),
    "Easy/10_Rainbow_Table": ("rainbow-table", "Easy"),
    "Easy/11_Matrix_Copy": ("matrix-copy", "Easy"),
    "Easy/12_Simple_Inference": ("simple-inference", "Easy"),
    "Easy/13_Count_Array_Element": ("count-array-element", "Easy"),
    "Easy/14_Count_2D_Array_Element": ("count-2d-array-element", "Easy"),
    "Easy/15_Sigmoid_Linear_Unit": ("sigmoid-linear-unit", "Easy"),
    "Easy/16_Swish-Gated_Linear_Unit": ("swish-gated-linear-unit", "Easy"),
    "Easy/17_Value_Clipping": ("value-clipping", "Easy"),
    "Easy/18_Interleave_Arrays": ("interleave-arrays", "Easy"),
    "Easy/19_Gaussian_Error_Gated_Linear_Unit": ("gaussian-error-gated-linear-unit", "Easy"),
    "Easy/20_RGB_to_Grayscale": ("rgb-to-grayscale", "Easy"),
    # Medium
    "Medium/21_Reduction": ("reduction", "Medium"),
    "Medium/22_Softmax": ("softmax", "Medium"),
    "Medium/23_Softmax_Attention": ("softmax-attention", "Medium"),
    "Medium/24_2D_Convolution": ("2d-convolution", "Medium"),
    "Medium/25_Histogramming": ("histogramming", "Medium"),
    "Medium/26_Sorting": ("sorting", "Medium"),
    "Medium/27_Prefix_Sum": ("prefix-sum", "Medium"),
    "Medium/28_Dot_Product": ("dot-product", "Medium"),
    "Medium/29_Sparse_Matrix-Vector_Multiplication": ("sparse-matrix-vector-multiplication", "Medium"),
    "Medium/30_General_Matrix_Multiplication_(GEMM)": ("general-matrix-multiplication-gemm", "Medium"),
    "Medium/31_Categorical_Cross_Entropy_Loss": ("categorical-cross-entropy-loss", "Medium"),
    "Medium/32_Mean_Squared_Error": ("mean-squared-error", "Medium"),
    "Medium/33_Gaussian_Blur": ("gaussian-blur", "Medium"),
    "Medium/34_Top_K_Selection": ("top-k-selection", "Medium"),
    "Medium/35_Batched_Matrix_Multiplication": ("batched-matrix-multiplication", "Medium"),
    "Medium/36_INT8_Quantized_MatMul": ("int8-quantized-matmul", "Medium"),
    "Medium/37_Ordinary_Least_Squares": ("ordinary-least-squares", "Medium"),
    "Medium/38_Logistic_Regression": ("logistic-regression", "Medium"),
    "Medium/39_Monte_Carlo_Integration": ("monte-carlo-integration", "Medium"),
    "Medium/40_Radix_Sort": ("radix-sort", "Medium"),
    "Medium/41_Matrix_Power": ("matrix-power", "Medium"),
    "Medium/42_Nearest_Neighbor": ("nearest-neighbor", "Medium"),
    "Medium/43_Batch_Normalization": ("batch-normalization", "Medium"),
    "Medium/44_2D_Max_Pooling": ("2d-max-pooling", "Medium"),
    "Medium/45_Count_3D_Array_Element": ("count-3d-array-element", "Medium"),
    "Medium/46_BFS_Shortest_Path": ("bfs-shortest-path", "Medium"),
    "Medium/47_Subarray_Sum": ("subarray-sum", "Medium"),
    "Medium/48_2D_Subarray_Sum": ("2d-subarray-sum", "Medium"),
    "Medium/49_3D_Subarray_Sum": ("3d-subarray-sum", "Medium"),
    "Medium/50_RMS_Normalization": ("rms-normalization", "Medium"),
    "Medium/51_Max_Subarray_Sum": ("max-subarray-sum", "Medium"),
    "Medium/52_Attention_with_Linear_Biases": ("attention-with-linear-biases", "Medium"),
    "Medium/53_FP16_Batched_Matrix_Multiplication": ("fp16-batched-matrix-multiplication", "Medium"),
    "Medium/54_FP16_Dot_Product": ("fp16-dot-product", "Medium"),
    "Medium/55_Top-p_Sampling": ("top-p-sampling", "Medium"),
    "Medium/56_Rotary_Positional_Embedding": ("rotary-positional-embedding", "Medium"),
    "Medium/57_Weight_Dequantization": ("weight-dequantization", "Medium"),
    "Medium/58_MoE_Top-K_Gating": ("moe-top-k-gating", "Medium"),
    # Hard
    "Hard/59_3D_Convolution": ("3d-convolution", "Hard"),
    "Hard/60_Multi-Head_Attention": ("multi-head-attention", "Hard"),
    "Hard/61_Multi-Agent_Simulation": ("multi-agent-simulation", "Hard"),
    "Hard/62_K-Means_Clustering": ("k-means-clustering", "Hard"),
    "Hard/63_Fast_Fourier_Transform": ("fast-fourier-transform", "Hard"),
    "Hard/64_Causal_Self-Attention": ("causal-self-attention", "Hard"),
    "Hard/65_Linear_Self-Attention": ("linear-self-attention", "Hard"),
    "Hard/66_Sliding_Window_Self-Attention": ("sliding-window-self-attention", "Hard"),
}

# Title mapping from API title to folder title
TITLE_TO_FOLDER = {}
for folder, (slug, diff) in FOLDER_TO_SLUG.items():
    # Extract just the title part from folder name (after number_)
    parts = folder.split("/")
    folder_name = parts[1]  # e.g., "05_Matrix_Addition"
    title_part = "_".join(folder_name.split("_")[1:])  # e.g., "Matrix_Addition"
    TITLE_TO_FOLDER[title_part.lower().replace("_", " ").replace("-", " ").replace("(", "").replace(")", "")] = folder


class HTMLToMarkdown:
    """Convert HTML spec to structured sections."""

    def __init__(self, html):
        self.html = html
        self.sections = {
            'description': [],
            'requirements': [],
            'examples': [],
            'constraints': []
        }
        self._parse()

    def _parse(self):
        # Remove leading/trailing whitespace from HTML
        html = self.html.strip()

        # Split by h2 tags
        # First, get everything before the first h2 as description
        parts = re.split(r'<h2>(.*?)</h2>', html, flags=re.DOTALL)

        if parts:
            # First part is description (before any h2)
            self.sections['description'] = self._html_to_text(parts[0])

        # Process h2 sections
        i = 1
        while i < len(parts) - 1:
            heading = parts[i].strip()
            content = parts[i + 1]

            if 'Implementation' in heading or 'Requirement' in heading:
                self.sections['requirements'] = self._extract_list_items(content)
            elif re.match(r'Example', heading):
                example_text = self._extract_pre(content)
                self.sections['examples'].append({
                    'heading': heading.rstrip(':'),
                    'content': example_text
                })
            elif 'Constraint' in heading:
                self.sections['constraints'] = self._extract_list_items(content)

            i += 2

    def _html_to_text(self, html):
        """Convert HTML paragraph to plain text."""
        text = html
        # Handle code tags
        text = re.sub(r'<code>(.*?)</code>', r'`\1`', text)
        # Handle math/katex - simplify
        text = re.sub(r'<span class="katex">.*?</span>', '', text)
        # Remove all remaining tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Fix common HTML entities
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        text = text.replace('&le;', '≤').replace('&ge;', '≥')
        text = text.replace('&times;', '×')
        return text

    def _extract_list_items(self, html):
        """Extract list items from HTML."""
        items = re.findall(r'<li>(.*?)</li>', html, re.DOTALL)
        return [self._html_to_text(item) for item in items]

    def _extract_pre(self, html):
        """Extract pre/code block content."""
        match = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
        if match:
            text = match.group(1)
            # Remove HTML tags inside pre
            text = re.sub(r'<[^>]+>', '', text)
            text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            return text.strip()
        # If no pre tag, get all text
        return self._html_to_text(html)


def fetch_json(url):
    """Fetch JSON from URL."""
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))


def find_challenge_by_title(challenges, folder_path):
    """Find a challenge in the API data that matches the folder."""
    folder_name = os.path.basename(folder_path)
    # Extract title from folder name: "05_Matrix_Addition" -> "Matrix Addition"
    title_from_folder = " ".join(folder_name.split("_")[1:])

    # Also handle special cases
    title_from_folder_lower = title_from_folder.lower()

    for c in challenges:
        api_title_lower = c['title'].lower()
        if api_title_lower == title_from_folder_lower:
            return c
        # Handle parentheses: "General Matrix Multiplication (GEMM)" vs "General_Matrix_Multiplication_(GEMM)"
        if api_title_lower.replace("(", "").replace(")", "") == title_from_folder_lower.replace("(", "").replace(")", ""):
            return c
        # Handle hyphens
        if api_title_lower.replace("-", " ") == title_from_folder_lower.replace("-", " "):
            return c

    return None


def get_cuda_starter_code(challenge_id):
    """Fetch the CUDA starter code for a challenge."""
    try:
        data = fetch_json(f"{API_BASE}/challenges/{challenge_id}/starter-code")
        for code in data.get('starterCode', []):
            if code.get('language') == 'cuda':
                return code.get('fileContent', '')
    except Exception as e:
        print(f"  Warning: Could not fetch starter code: {e}")
    return ''


def generate_readme(title, slug, difficulty, spec_html, code_template):
    """Generate README.md content from challenge data."""
    parser = HTMLToMarkdown(spec_html)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> LeetGPU: https://leetgpu.com/challenges/{slug}")
    lines.append("")
    lines.append("## 难度")
    lines.append("")
    lines.append(difficulty)
    lines.append("")
    lines.append("## 题目描述")
    lines.append("")
    lines.append(parser.sections['description'])
    lines.append("")

    if parser.sections['requirements']:
        lines.append("## 实现要求")
        lines.append("")
        for req in parser.sections['requirements']:
            lines.append(f"- {req}")
        lines.append("")

    if parser.sections['examples']:
        lines.append("## 示例")
        lines.append("")
        for ex in parser.sections['examples']:
            lines.append(f"**{ex['heading']}：**")
            lines.append("")
            lines.append("```")
            lines.append(ex['content'])
            lines.append("```")
            lines.append("")

    if parser.sections['constraints']:
        lines.append("## 约束条件")
        lines.append("")
        for con in parser.sections['constraints']:
            lines.append(f"- {con}")
        lines.append("")

    if code_template:
        lines.append("## 代码模板")
        lines.append("")
        lines.append("```cpp")
        lines.append(code_template.rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    print("Fetching challenges from API...")
    data = fetch_json(f"{API_BASE}/challenges")
    challenges = data['challenges']
    print(f"Found {len(challenges)} challenges")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for folder_path, (slug, difficulty) in sorted(FOLDER_TO_SLUG.items()):
        full_path = os.path.join(BASE_DIR, folder_path, "README.md")
        folder_name = os.path.basename(folder_path)
        title_display = " ".join(folder_name.split("_")[1:])

        print(f"\nProcessing: {folder_path}")

        # Find matching challenge in API data
        challenge = find_challenge_by_title(challenges, folder_path)

        if not challenge:
            print(f"  ERROR: Could not find matching challenge for '{title_display}'")
            fail_count += 1
            continue

        print(f"  Matched API: id={challenge['id']}, title='{challenge['title']}'")

        # Fetch starter code
        code_template = get_cuda_starter_code(challenge['id'])

        # Generate README
        readme_content = generate_readme(
            title=challenge['title'],
            slug=slug,
            difficulty=difficulty,
            spec_html=challenge['spec'],
            code_template=code_template
        )

        # Write file
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"  Written: {full_path}")
        success_count += 1

    print(f"\n{'='*60}")
    print(f"Done! Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")
    print(f"Total processed: {success_count + skip_count + fail_count}")


if __name__ == "__main__":
    main()
