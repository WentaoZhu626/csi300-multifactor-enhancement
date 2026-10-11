# CSI 300 Multifactor Index Enhancement

沪深300多因子指数增强模型的个人研报复现与修改版本。

- `src/seven_modified.py`：修改后的主要实现。
- `src/seven_original.py`：保留的早期实现，便于比较。
- `references/csi300_multifactor_index_enhancement.pdf`：对应的多因子研报。
- `docs/csi300_model_explanation.docx`：作者的模型解释文档。

实现包含因子构建、截面预处理、行业与风险处理、因子检验及组合优化等步骤。具体流程以代码与解释文档为准。

代码依赖聚宽 `jqdata` 平台环境及其数据接口。`requirements.txt` 仅列出普通 Python 科学计算依赖，不将 `jqdata` 当作可直接用 pip 安装的替代数据服务。运行需要在相应平台中检查日期、证券池和参数；本次发布没有重新运行平台回测。

原研报的作者与版权归原发布机构。个人实现与原研报一并保存用于方法对照，未为第三方材料附加新的许可证。
