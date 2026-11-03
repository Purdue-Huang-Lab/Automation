"""Read-only table model over numpy columns (fast for the ~10k-50k row combination list)."""

from typing import Callable, List, Optional, Sequence

from PyQt5 import QtCore, QtGui


class ArrayTableModel(QtCore.QAbstractTableModel):
    """Columns of equal length, each formatted on demand by its ``fmt`` callable."""

    def __init__(self, headers: Sequence[str], fmts: Sequence[Callable[[object], str]], parent=None):
        super().__init__(parent)
        self._headers = list(headers)
        self._fmts = list(fmts)
        self._cols: List[Sequence] = [[] for _ in headers]
        self._rows = 0
        self._row_bg: Optional[Sequence[Optional[QtGui.QBrush]]] = None

    def set_columns(self, cols: Sequence[Sequence], row_bg: Optional[Sequence[Optional[QtGui.QBrush]]] = None) -> None:
        self.beginResetModel()
        self._cols = list(cols)
        self._rows = len(cols[0]) if cols else 0
        self._row_bg = row_bg
        self.endResetModel()

    def clear(self) -> None:
        self.set_columns([[] for _ in self._headers])

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role == QtCore.Qt.DisplayRole:
            return self._fmts[c](self._cols[c][r])
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if role == QtCore.Qt.BackgroundRole and self._row_bg is not None:
            return self._row_bg[r]
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self._headers[section]
        return str(section + 1)
