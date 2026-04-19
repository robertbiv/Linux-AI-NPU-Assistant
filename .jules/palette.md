## 2024-04-17 - Add AccessibleName to PyQt buttons
**Learning:** QToolButton and icon-only QPushButton in PyQt need setAccessibleName to be read properly by screen readers, even when setToolTip is provided. While some screen readers fallback to tooltips, setting accessible name is the standard and most robust way to ensure accessibility for icon-only buttons.
**Action:** Add setAccessibleName to all icon-only buttons in the GUI, mirroring their tooltips, to ensure proper accessibility.

## 2024-04-18 - Add AccessibleName to PyQt Item Widgets
**Learning:** In PyQt5, container widgets that display items, such as `QListWidget` and `QTableWidget`, need an explicit `setAccessibleName()` called on the parent widget instance itself to ensure screen readers announce the overall purpose of the container when it receives focus, beyond just reading the individual items.
**Action:** When adding accessibility to item-based widgets, set `setAccessibleName` on the main widget (e.g., `self._list.setAccessibleName("Available Models")` and `table.setAccessibleName("Diagnostic Status Table")`).
