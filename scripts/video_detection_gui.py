#!/usr/bin/env python3
"""PyQt5 interface for selecting images/videos and viewing YOLOv7 results."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import QProcess, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QComboBox,
    QSlider,
    QSplitter,
    QScrollArea,
)

from scripts.batch_video_detect import VIDEO_SUFFIXES as BATCH_VIDEO_SUFFIXES, safe_name


PATH_ROLE = Qt.UserRole
IS_DIRECTORY_ROLE = Qt.UserRole + 1
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo"}
VIDEO_SUFFIXES = BATCH_VIDEO_SUFFIXES - {".asf", ".webm"}  # unsupported by this YOLOv7 loader


class MediaTreeWidget(QTreeWidget):
    mediaDropped = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DropOnly)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: Any) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.mediaDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class VideoDetectionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Logo Detection Viewer")
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            initial_width = max(1, min(1280, int(available.width() * 0.94)))
            initial_height = max(1, min(960, int(available.height() * 0.92)))
            self.setMinimumSize(min(720, initial_width), min(540, initial_height))
            self.resize(initial_width, initial_height)
            self.move(
                available.x() + max(0, (available.width() - initial_width) // 2),
                available.y() + max(0, (available.height() - initial_height) // 2),
            )
        else:
            self.resize(1100, 760)
            self.setMinimumSize(720, 540)

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.batch_dir: Path | None = None
        self.queue: list[Path] = []
        self.current_media: Path | None = None
        self.current_output: Path | None = None
        self.results: list[dict[str, Any]] = []
        self.cancel_requested = False
        self.current_completed = False
        self.video_capture: cv2.VideoCapture | None = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._advance_video)
        self.current_pixmap: QPixmap | None = None
        self.video_fps = 25.0

        self._build_ui()
        self._refresh_tree()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(14)
        self.main_splitter = QSplitter(Qt.Vertical)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidget(controls_widget)
        self.main_splitter.addWidget(controls_scroll)

        header = QFrame()
        header.setObjectName("Hero")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 18, 16)
        title_column = QVBoxLayout()
        eyebrow = QLabel("COMPUTER VISION WORKSPACE")
        eyebrow.setObjectName("Eyebrow")
        title = QLabel("Logo Detection Studio")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("Select media, run YOLOv7, and review the annotated results.")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        title_column.addWidget(eyebrow)
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        title_column.addStretch(1)
        header_layout.addLayout(title_column, 1)
        artwork = QLabel()
        artwork.setAlignment(Qt.AlignCenter)
        artwork.setFixedSize(260, 122)
        artwork_path = PROJECT_ROOT / "assets" / "logo_detection_header.png"
        if artwork_path.is_file():
            artwork.setPixmap(QPixmap(str(artwork_path)).scaled(260, 122, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header_layout.addWidget(artwork)
        controls_layout.addWidget(header)

        media_group = QGroupBox("1  ·  Select images or videos")
        media_layout = QVBoxLayout(media_group)
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Dataset"))
        self.dataset_mode = "videos"
        self.dropped_group: QTreeWidgetItem | None = None
        self.dataset_buttons = QButtonGroup(self)
        self.dataset_buttons.setExclusive(True)
        self.videos_button = QPushButton("🎬  Videos")
        self.videos_button.setCheckable(True)
        self.videos_button.setObjectName("DatasetSelector")
        self.images_button = QPushButton("▧  Images")
        self.images_button.setCheckable(True)
        self.images_button.setObjectName("DatasetSelector")
        self.dataset_buttons.addButton(self.videos_button)
        self.dataset_buttons.addButton(self.images_button)
        self.videos_button.setChecked(True)
        self.videos_button.setMinimumWidth(112)
        self.images_button.setMinimumWidth(112)
        self.videos_button.clicked.connect(lambda: self._switch_dataset("videos"))
        self.images_button.clicked.connect(lambda: self._switch_dataset("images"))
        root_row.addWidget(self.videos_button)
        root_row.addWidget(self.images_button)
        self.root_edit = QLineEdit(str(PROJECT_ROOT / "test_videos"))
        root_row.addWidget(self.root_edit, 1)
        self.root_browse = QPushButton("Browse…")
        self.root_browse.clicked.connect(self._browse_root)
        root_row.addWidget(self.root_browse)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh_tree)
        root_row.addWidget(self.refresh_button)
        media_layout.addLayout(root_row)

        self.tree = MediaTreeWidget()
        self.tree.setHeaderLabels(["Folders and media", "Type"])
        self.tree.setAlternatingRowColors(True)
        self.tree.itemChanged.connect(self._tree_item_changed)
        self.tree.mediaDropped.connect(self._add_dropped_media)
        media_layout.addWidget(self.tree)
        drop_hint = QLabel("Tip: drag image/video files or folders from your file manager into this list.")
        drop_hint.setObjectName("Hint")
        media_layout.addWidget(drop_hint)
        controls_layout.addWidget(media_group, 2)

        settings_group = QGroupBox("2  ·  Model and detection settings")
        settings = QFormLayout()
        settings.setHorizontalSpacing(16)
        settings.setVerticalSpacing(10)
        weights_row = QHBoxLayout()
        self.weights_edit = QLineEdit(str(PROJECT_ROOT / "runs/train/logo_single_microsoft_v3/weights/best.pt"))
        weights_row.addWidget(self.weights_edit, 1)
        self.weights_browse = QPushButton("Browse…")
        self.weights_browse.clicked.connect(self._browse_weights)
        weights_row.addWidget(self.weights_browse)
        settings.addRow("YOLO checkpoint", weights_row)

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(PROJECT_ROOT / "runs/batch_video_detect"))
        output_row.addWidget(self.output_edit, 1)
        self.output_browse = QPushButton("Browse…")
        self.output_browse.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_browse)
        settings.addRow("Results parent folder", output_row)

        inference_row = QHBoxLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.4)
        inference_row.addWidget(QLabel("Confidence"))
        inference_row.addWidget(self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        inference_row.addWidget(QLabel("IoU"))
        inference_row.addWidget(self.iou_spin)
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 4096)
        self.img_size_spin.setSingleStep(32)
        self.img_size_spin.setValue(640)
        inference_row.addWidget(QLabel("Image size"))
        inference_row.addWidget(self.img_size_spin)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["0", "cpu"])
        inference_row.addWidget(QLabel("Device"))
        inference_row.addWidget(self.device_combo)
        inference_row.addStretch(1)
        settings.addRow("Detection settings", inference_row)

        self.save_labels_check = QCheckBox("Save detection labels and confidence scores")
        self.save_labels_check.setChecked(True)
        settings.addRow("Outputs", self.save_labels_check)
        settings_group.setLayout(settings)
        controls_layout.addWidget(settings_group)

        run_group = QGroupBox("3  ·  Run")
        run_layout = QVBoxLayout(run_group)
        action_row = QHBoxLayout()
        self.selection_label = QLabel("0 items selected")
        action_row.addWidget(self.selection_label, 1)
        self.start_button = QPushButton("Start detection")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._start_batch)
        action_row.addWidget(self.start_button)
        self.cancel_button = QPushButton("Cancel batch")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_batch)
        action_row.addWidget(self.cancel_button)
        run_layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        run_layout.addWidget(self.progress)
        controls_layout.addWidget(run_group)
        controls_layout.addStretch(1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Detection output and errors will appear here.")

        result_splitter = QSplitter(Qt.Horizontal)
        self.result_list = QListWidget()
        self.result_list.currentItemChanged.connect(self._show_selected_result)
        result_splitter.addWidget(self.result_list)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.preview_label = QLabel("Completed detections will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(420, 260)
        self.preview_label.setStyleSheet("QLabel { background-color: #20242b; color: #e8e8e8; }")
        preview_layout.addWidget(self.preview_label, 1)
        video_controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_video)
        video_controls.addWidget(self.play_button)
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.sliderMoved.connect(self._seek_video)
        video_controls.addWidget(self.video_slider, 1)
        self.video_time_label = QLabel("00:00 / 00:00")
        video_controls.addWidget(self.video_time_label)
        preview_layout.addLayout(video_controls)
        result_splitter.addWidget(preview_panel)
        result_splitter.setStretchFactor(1, 1)

        result_panel = QSplitter(Qt.Vertical)
        result_panel.addWidget(result_splitter)
        result_panel.addWidget(self.log)
        result_panel.setStretchFactor(0, 3)
        result_panel.setStretchFactor(1, 1)
        result_group = QGroupBox("4  ·  Detection results")
        result_layout = QVBoxLayout(result_group)
        result_layout.addWidget(result_panel)
        self.main_splitter.addWidget(result_group)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([500, 350])
        layout.addWidget(self.main_splitter)

        self.setCentralWidget(root)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0b111b; color: #e7edf7; font-family: 'Segoe UI', 'Noto Sans', sans-serif; font-size: 10pt; }
            QFrame#Hero { background-color: #111d2b; border: 1px solid #1d3447; border-radius: 16px; }
            QLabel#Eyebrow { color: #53d6bd; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; }
            QLabel#HeroTitle { color: #f4f7fb; font-size: 23pt; font-weight: 700; }
            QLabel#HeroSubtitle { color: #a8b8c9; font-size: 10pt; }
            QGroupBox { background-color: #111925; border: 1px solid #263344; border-radius: 12px; margin-top: 12px; padding: 14px 12px 12px 12px; font-weight: 650; color: #dce8f2; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #75d9c5; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTreeWidget, QListWidget { background-color: #0b121d; border: 1px solid #2b394a; border-radius: 7px; padding: 6px; selection-background-color: #176d72; selection-color: #ffffff; }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus { border: 1px solid #4dcab5; }
            QTreeWidget, QListWidget { alternate-background-color: #101925; }
            QHeaderView::section { background-color: #172333; color: #b8c9d8; border: 0; border-bottom: 1px solid #2b394a; padding: 7px; font-weight: 600; }
            QPushButton { background-color: #1a2939; border: 1px solid #34475b; border-radius: 7px; padding: 8px 13px; color: #e9f2fa; font-weight: 600; }
            QPushButton:hover { background-color: #23384b; border-color: #4abda9; }
            QPushButton:pressed { background-color: #145b60; }
            QPushButton:disabled { color: #708093; background-color: #141d28; border-color: #283342; }
            QPushButton#Primary { background-color: #137b73; border-color: #2aa995; color: white; padding: 9px 17px; }
            QPushButton#Primary:hover { background-color: #199589; }
            QPushButton#DatasetSelector { padding: 9px 14px; }
            QPushButton#DatasetSelector:checked { background-color: #145f65; border: 1px solid #54d6bd; color: #ffffff; }
            QSplitter::handle:vertical { height: 8px; background-color: #202e3d; }
            QSplitter::handle:horizontal { width: 8px; background-color: #202e3d; }
            QSplitter::handle:hover { background-color: #39b8a4; }
            QScrollArea { border: 0; background: transparent; }
            QProgressBar { background-color: #0a111a; border: 1px solid #2b394a; border-radius: 6px; height: 15px; text-align: center; color: #dce8f2; }
            QProgressBar::chunk { background-color: #39b8a4; border-radius: 5px; }
            QSlider::groove:horizontal { height: 5px; background: #2b394a; border-radius: 2px; }
            QSlider::handle:horizontal { background: #53d6bd; border: 1px solid #9bf1df; width: 13px; margin: -5px 0; border-radius: 7px; }
            QCheckBox { spacing: 8px; color: #bfcbd8; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QLabel { background: transparent; }
        """)

    def _browse_root(self) -> None:
        current = self.root_edit.text().strip() or str(PROJECT_ROOT)
        folder = QFileDialog.getExistingDirectory(self, "Select media folder", current)
        if folder:
            self.root_edit.setText(folder)
            self._refresh_tree()

    def _switch_dataset(self, mode: str) -> None:
        if mode == self.dataset_mode:
            return
        self.dataset_mode = mode
        dataset_name = "test_images" if mode == "images" else "test_videos"
        self.root_edit.setText(str(PROJECT_ROOT / dataset_name))
        self._refresh_tree()

    def _browse_weights(self) -> None:
        current = self.weights_edit.text().strip() or str(PROJECT_ROOT)
        filename, _ = QFileDialog.getOpenFileName(self, "Select YOLO checkpoint", current, "PyTorch checkpoints (*.pt);;All files (*)")
        if filename:
            self.weights_edit.setText(filename)

    def _browse_output(self) -> None:
        current = self.output_edit.text().strip() or str(PROJECT_ROOT / "runs")
        folder = QFileDialog.getExistingDirectory(self, "Select results parent folder", current)
        if folder:
            self.output_edit.setText(folder)

    def _refresh_tree(self) -> None:
        root_path = Path(self.root_edit.text()).expanduser().resolve()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.dropped_group = None
        if not root_path.is_dir():
            self.tree.blockSignals(False)
            self._update_selection_label()
            QMessageBox.warning(self, "Folder not found", f"The media folder does not exist:\n{root_path}")
            return
        root_item = self._make_item(root_path, is_directory=True)
        self.tree.addTopLevelItem(root_item)
        self._populate_directory(root_item, root_path)
        root_item.setExpanded(True)
        self.tree.blockSignals(False)
        self._update_selection_label()

    def _make_item(self, path: Path, is_directory: bool) -> QTreeWidgetItem:
        if is_directory:
            kind = "Folder"
        else:
            kind = "Image" if self.dataset_mode == "images" else "Video"
        item = QTreeWidgetItem([path.name or str(path), kind])
        item.setData(0, PATH_ROLE, str(path))
        item.setData(0, IS_DIRECTORY_ROLE, is_directory)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setCheckState(0, Qt.Unchecked)
        return item

    def _populate_directory(self, parent: QTreeWidgetItem, directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower()))
        except OSError as exc:
            self._append_log(f"Cannot read folder {directory}: {exc}\n")
            return
        for path in entries:
            if path.is_dir():
                child = self._make_item(path, is_directory=True)
                parent.addChild(child)
                self._populate_directory(child, path)
            elif path.is_file() and path.suffix.lower() in self._allowed_suffixes():
                parent.addChild(self._make_item(path, is_directory=False))

    def _allowed_suffixes(self) -> set[str]:
        return IMAGE_SUFFIXES if self.dataset_mode == "images" else VIDEO_SUFFIXES

    def _add_dropped_media(self, paths: list[Path]) -> None:
        accepted: list[Path] = []
        ignored: list[Path] = []
        suffixes = self._allowed_suffixes()
        for path in paths:
            path = path.expanduser().resolve()
            if path.is_dir():
                accepted.extend(
                    candidate for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in suffixes
                )
            elif path.is_file() and path.suffix.lower() in suffixes:
                accepted.append(path)
            else:
                ignored.append(path)

        if not accepted:
            self._append_log("The dragged items contained no supported media for the selected dataset.\n")
            return

        if self.dropped_group is None:
            self.tree.blockSignals(True)
            self.dropped_group = self._make_item(PROJECT_ROOT / "Dragged media", is_directory=True)
            self.dropped_group.setText(1, "Added by drag and drop")
            self.tree.addTopLevelItem(self.dropped_group)
            self.dropped_group.setExpanded(True)
            self.tree.blockSignals(False)

        existing = {
            Path(self.dropped_group.child(i).data(0, PATH_ROLE))
            for i in range(self.dropped_group.childCount())
        }
        self.tree.blockSignals(True)
        added = 0
        for path in sorted(set(accepted)):
            if path in existing:
                continue
            if path.is_dir():
                item = self._make_item(path, is_directory=True)
                self.dropped_group.addChild(item)
                self._populate_directory(item, path)
            else:
                item = self._make_item(path, is_directory=False)
                self.dropped_group.addChild(item)
                existing.add(path)
            item.setCheckState(0, Qt.Checked)
            self._set_descendants(item, Qt.Checked)
            added += 1
        self.dropped_group.setExpanded(True)
        self.tree.blockSignals(False)
        self._update_ancestors(self.dropped_group)
        self._update_selection_label()
        self._append_log(f"Added {added} dragged media item(s) to the selection.\n")
        if ignored:
            self._append_log(f"Ignored {len(ignored)} unsupported or missing item(s).\n")

    @staticmethod
    def _relative_media_path(media: Path, root: Path) -> Path:
        try:
            return media.relative_to(root)
        except ValueError:
            parent_hash = hashlib.sha1(str(media.parent).encode("utf-8")).hexdigest()[:8]
            return Path("_dragged_media") / safe_name(media.parent.name) / parent_hash / media.name

    @staticmethod
    def _summary_media_path(media: Path, root: Path) -> str:
        try:
            return str(media.relative_to(root))
        except ValueError:
            return str(media)

    def _tree_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        self.tree.blockSignals(True)
        is_directory = bool(item.data(0, IS_DIRECTORY_ROLE))
        if is_directory:
            self._set_descendants(item, item.checkState(0))
        self._update_ancestors(item.parent())
        self.tree.blockSignals(False)
        self._update_selection_label()

    def _set_descendants(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            child.setCheckState(0, state)
            self._set_descendants(child, state)

    def _update_ancestors(self, item: QTreeWidgetItem | None) -> None:
        while item is not None:
            states = [item.child(i).checkState(0) for i in range(item.childCount())]
            if states and all(state == Qt.Checked for state in states):
                item.setCheckState(0, Qt.Checked)
            elif not states or all(state == Qt.Unchecked for state in states):
                item.setCheckState(0, Qt.Unchecked)
            else:
                item.setCheckState(0, Qt.PartiallyChecked)
            item = item.parent()

    def _selected_media(self) -> list[Path]:
        selected: list[Path] = []

        def walk(item: QTreeWidgetItem) -> None:
            if not item.data(0, IS_DIRECTORY_ROLE) and item.checkState(0) == Qt.Checked:
                selected.append(Path(item.data(0, PATH_ROLE)))
            for index in range(item.childCount()):
                walk(item.child(index))

        for index in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(index))
        return sorted(set(selected))

    def _update_selection_label(self) -> None:
        count = len(self._selected_media())
        self.selection_label.setText(f"{count} item{'s' if count != 1 else ''} selected")

    def _start_batch(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            return
        weights = Path(self.weights_edit.text()).expanduser().resolve()
        output_parent = Path(self.output_edit.text()).expanduser().resolve()
        selected = self._selected_media()
        if not weights.is_file():
            QMessageBox.warning(self, "Checkpoint not found", f"Select an existing YOLO checkpoint:\n{weights}")
            return
        if not selected:
            QMessageBox.information(self, "No media selected", "Check an image, video, or folder in the media tree first.")
            return
        if not output_parent.is_dir():
            answer = QMessageBox.question(
                self,
                "Create results folder?",
                f"This folder does not exist:\n{output_parent}\n\nCreate it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return
            output_parent.mkdir(parents=True, exist_ok=True)

        root = Path(self.root_edit.text()).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = output_parent / timestamp
        suffix = 1
        while batch_dir.exists():
            batch_dir = output_parent / f"{timestamp}_{suffix}"
            suffix += 1
        batch_dir.mkdir(parents=True)

        self.batch_dir = batch_dir
        self.queue = selected
        self.results = []
        self.cancel_requested = False
        self.current_media = None
        self.current_output = None
        self.current_completed = False
        self.progress.setRange(0, len(selected))
        self.progress.setValue(0)
        self.log.clear()
        self.result_list.clear()
        self._stop_video_preview()
        self._append_log(f"Starting batch of {len(selected)} item(s).\nResults: {batch_dir}\n\n")
        self._set_running(True)
        self._start_next_video()

    def _start_next_video(self) -> None:
        if not self.queue or self.cancel_requested:
            self._finish_batch()
            return
        assert self.batch_dir is not None
        self.current_media = self.queue.pop(0)
        self.current_completed = False
        root = Path(self.root_edit.text()).expanduser().resolve()
        relative = self._relative_media_path(self.current_media, root)
        run_parts = tuple(safe_name(part) for part in relative.parent.parts if part != ".")
        run_name = Path(*run_parts, safe_name(self.current_media.stem))
        self.current_output = self.batch_dir / run_name
        self.current_output.parent.mkdir(parents=True, exist_ok=True)

        detect_script = PROJECT_ROOT / "yolov7" / "detect.py"
        args = [
            str(detect_script),
            f"--weights={Path(self.weights_edit.text()).expanduser().resolve()}",
            f"--source={self.current_media}",
            f"--conf-thres={self.conf_spin.value()}",
            f"--iou-thres={self.iou_spin.value()}",
            f"--img-size={self.img_size_spin.value()}",
            f"--project={self.batch_dir}",
            f"--name={run_name.as_posix()}",
            f"--device={self.device_combo.currentText()}",
            "--save-conf",
            "--exist-ok",
        ]
        if self.save_labels_check.isChecked():
            args.append("--save-txt")

        index = len(self.results) + 1
        total = index + len(self.queue)
        self._append_log(f"[{index}/{total}] Detecting {relative}\n")
        self.process.start(sys.executable, args)

    def _read_stdout(self) -> None:
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(output)

    def _read_stderr(self) -> None:
        output = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._append_log(output)

    def _process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self.current_completed or self.current_media is None:
            return
        status = "cancelled" if self.cancel_requested else ("success" if exit_code == 0 else "failed")
        self._complete_current(status, exit_code)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.FailedToStart and not self.current_completed and self.current_media is not None:
            self._append_log("Could not start the detector process. Check Python and the YOLOv7 environment.\n")
            self._complete_current("failed", -1)

    def _complete_current(self, status: str, return_code: int) -> None:
        if self.current_completed or self.current_media is None:
            return
        self.current_completed = True
        root = Path(self.root_edit.text()).expanduser().resolve()
        output_file = self.current_output / self.current_media.name if self.current_output else None
        self.results.append({
            "media": self._summary_media_path(self.current_media, root),
            "status": status,
            "return_code": return_code,
            "output_dir": str(self.current_output) if self.current_output else "",
            "output_file": str(output_file) if output_file and output_file.is_file() else "",
        })
        if status == "success" and output_file and output_file.is_file():
            result_item = QListWidgetItem(f"{self.current_media.name} — detected")
            result_item.setData(Qt.UserRole, str(output_file))
            result_item.setData(Qt.UserRole + 1, self.current_media.suffix.lower() in VIDEO_SUFFIXES)
            self.result_list.addItem(result_item)
        self.progress.setValue(len(self.results))
        if status != "success":
            self._append_log(f"Finished {status}: {self.current_media.name} (exit {return_code})\n")
        self.current_media = None
        self.current_output = None
        if self.cancel_requested:
            for video in self.queue:
                self.results.append({
                    "media": self._summary_media_path(video, root),
                    "status": "cancelled",
                    "return_code": 130,
                    "output_dir": "",
                    "output_file": "",
                })
            self.queue.clear()
            self.progress.setValue(len(self.results))
            self._finish_batch()
        else:
            QTimer.singleShot(0, self._start_next_video)

    def _cancel_batch(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            self.cancel_requested = True
            root = Path(self.root_edit.text()).expanduser().resolve()
            for video in self.queue:
                self.results.append({
                    "media": self._summary_media_path(video, root),
                    "status": "cancelled",
                    "return_code": 130,
                    "output_dir": "",
                    "output_file": "",
                })
            self.queue.clear()
            self.progress.setValue(len(self.results))
            self._finish_batch()
            return
        self.cancel_requested = True
        self.cancel_button.setEnabled(False)
        self._append_log("Cancellation requested; stopping the current video…\n")
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.cancel_requested and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _finish_batch(self) -> None:
        if self.batch_dir is not None:
            (self.batch_dir / "summary.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")
            with (self.batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["media", "status", "return_code", "output_dir", "output_file"])
                writer.writeheader()
                writer.writerows(self.results)
            successes = sum(result["status"] == "success" for result in self.results)
            failures = sum(result["status"] not in ("success", "cancelled") for result in self.results)
            self._append_log(
                f"\nBatch finished: {successes} succeeded, {failures} failed, "
                f"{len(self.results) - successes - failures} cancelled.\n"
                f"Summary: {self.batch_dir / 'summary.csv'}\n"
            )
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.tree.setEnabled(not running)
        self.videos_button.setEnabled(not running)
        self.images_button.setEnabled(not running)
        self.root_edit.setEnabled(not running)
        self.root_browse.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.weights_edit.setEnabled(not running)
        self.weights_browse.setEnabled(not running)
        self.output_edit.setEnabled(not running)
        self.output_browse.setEnabled(not running)
        self.conf_spin.setEnabled(not running)
        self.iou_spin.setEnabled(not running)
        self.img_size_spin.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.save_labels_check.setEnabled(not running)

    def _show_selected_result(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self._stop_video_preview()
        self.current_pixmap = None
        if current is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Completed detections will appear here")
            return

        result_path = Path(current.data(Qt.UserRole))
        is_video = bool(current.data(Qt.UserRole + 1))
        if is_video:
            self.video_capture = cv2.VideoCapture(str(result_path))
            if not self.video_capture.isOpened():
                self.video_capture.release()
                self.video_capture = None
                self.preview_label.setText(f"Could not open video:\n{result_path}")
                return
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.video_fps = fps if fps > 0 else 25.0
            self.video_slider.setRange(0, max(0, frame_count - 1))
            self.video_slider.setEnabled(frame_count > 1)
            self.play_button.setEnabled(frame_count > 1)
            self._render_video_frame()
            self.video_timer.start(max(10, int(1000 / self.video_fps)))
            self.play_button.setText("Pause")
        else:
            self.current_pixmap = QPixmap(str(result_path))
            if self.current_pixmap.isNull():
                self.preview_label.setText(f"Could not load image:\n{result_path}")
                return
            self._scale_preview()
            self.video_slider.setEnabled(False)
            self.play_button.setEnabled(False)
            self.video_time_label.setText("Image")

    def _stop_video_preview(self) -> None:
        self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.play_button.setText("Play")
        self.play_button.setEnabled(False)
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        self.video_time_label.setText("00:00 / 00:00")

    def _render_video_frame(self) -> None:
        if self.video_capture is None:
            return
        ok, frame = self.video_capture.read()
        if not ok:
            self.video_timer.stop()
            self.play_button.setText("Play")
            return
        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(image)
        self._scale_preview()
        if not self.video_slider.isSliderDown():
            self.video_slider.setValue(max(0, current_frame))
        frame_count = max(0, int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        current_seconds = max(0, current_frame) / self.video_fps
        total_seconds = frame_count / self.video_fps
        self.video_time_label.setText(f"{self._format_time(current_seconds)} / {self._format_time(total_seconds)}")

    def _advance_video(self) -> None:
        self._render_video_frame()

    def _seek_video(self, frame_number: int) -> None:
        if self.video_capture is None:
            return
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._render_video_frame()

    def _toggle_video(self) -> None:
        if self.video_capture is None:
            return
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.play_button.setText("Play")
            return
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) >= frame_count:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_timer.start(max(10, int(1000 / self.video_fps)))
        self.play_button.setText("Pause")

    @staticmethod
    def _format_time(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def _scale_preview(self) -> None:
        if self.current_pixmap is not None and not self.current_pixmap.isNull():
            self.preview_label.setText("")
            self.preview_label.setPixmap(
                self.current_pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if hasattr(self, "preview_label"):
            self._scale_preview()

    def _append_log(self, text: str) -> None:
        if text:
            self.log.moveCursor(QTextCursor.End)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()

    def closeEvent(self, event: Any) -> None:
        self._stop_video_preview()
        if self.process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(
                self,
                "Detection is running",
                "Close the window and stop the current detection?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = VideoDetectionWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
