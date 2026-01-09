# src/notifications.py
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QColor, QFont, QPixmap, QPainter, QBrush

class RoundIcon(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setText(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Segoe UI Emoji", 20))
        self.setStyleSheet("""
            QLabel {
                background-color: #333333;
                color: white;
                border-radius: 20px;
            }
        """)

class DynamicIsland(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. Window Flags for Top-Most, Frameless Popup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool |
            Qt.WindowType.ToolTip 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Size
        self.width = 380
        self.height = 80
        self.resize(self.width, self.height)

        # Styling: macOS-like light, semi-transparent card
        self.setStyleSheet("""
            QWidget#Container {
                background-color: rgba(240, 240, 240, 0.95);
                border-radius: 18px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
            QLabel {
                color: black;
                background-color: transparent;
                border: none;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QLabel#Title {
                font-weight: bold;
                font-size: 13px;
            }
            QLabel#Message {
                font-size: 13px;
                color: #333333;
            }
        """)

        # Main layout for the widget
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Container Frame for styling
        self.container = QFrame()
        self.container.setObjectName("Container")
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(12, 12, 12, 12)
        container_layout.setSpacing(12)

        # Icon
        self.icon_label = RoundIcon("🛡️")
        container_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignTop)

        # Text Content (Title + Message)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.title_label = QLabel("ShieldAI Protection")
        self.title_label.setObjectName("Title")
        
        self.message_label = QLabel("Threat Detected")
        self.message_label.setObjectName("Message")
        self.message_label.setWordWrap(True)
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.message_label)
        container_layout.addLayout(text_layout)

        main_layout.addWidget(self.container)

        # 2. Subtle Drop Shadow
        shadow = QGraphicsDropShadowEffect(self.container)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 4)
        self.container.setGraphicsEffect(shadow)

        # Animation Setup
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide_notification)

    def show_message(self, title, message, color="#ff4d4d"):
        # Position: Top Right corner
        screen = self.screen().availableGeometry()
        x_pos = screen.width() - self.width - 20 # 20px padding from right
        y_start = -self.height - 20
        y_end = 40 # 40px from top

        # Update Text
        self.title_label.setText(title)
        #Truncate long messages
        if len(message) > 90: message = message[:87] + "..."
        self.message_label.setText(message)
        
        # Force to Front
        self.show()
        self.raise_()
        self.activateWindow()

        # Animate In
        self.animation.setStartValue(QPoint(x_pos, y_start))
        self.animation.setEndValue(QPoint(x_pos, y_end))
        self.animation.start()
        
        # Duration: 8 seconds
        self.timer.start(8000)

    def hide_notification(self):
        screen = self.screen().availableGeometry()
        x_pos = screen.width() - self.width - 20
        
        # Animate Out
        self.animation.setStartValue(self.pos())
        self.animation.setEndValue(QPoint(x_pos, -self.height - 20))
        self.animation.start()
        QTimer.singleShot(500, self.close)