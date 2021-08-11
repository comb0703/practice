import sys

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

import sys
from CURDsdk import CURD_sdk
from CURD_css import CURD_style

class VideoObj(QtCore.QObject):
    RGB_signal = QtCore.pyqtSignal(QtGui.QImage)
    complete_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(VideoObj, self).__init__(parent)
        self.CURD = CURD_sdk()

        self.top_w = 896
        self.top_h = 504

        self.is_running = False
        self.status = 0

    def sdk_init(self):
        return self.CURD.init_sdk()

    def is_sdk_init_done(self):
        return self.CURD.is_init_done()

    def frame_to_qimg(self, img_data, w, h):
        q_image = QtGui.QImage(img_data.data, w, h, img_data.strides[0], QtGui.QImage.Format_RGB888)
        return q_image

    def changeStatus(self, status):
        self.status = status

    @QtCore.pyqtSlot()
    def start_cam(self):

        # program error handling
        if not self.is_sdk_init_done():
            QtWidgets.QMessageBox.warning(self, "INIT ERROR", "Check Init status then restart this program.",\
                QtWidgets.QMessageBox.Ok)
            return False

        while self.is_running:
        
            # get frame
            c_frame = self.CURD.get_frame(self.top_w, self.top_h)
                      
            # draw Frame on win
            top_img = self.frame_to_qimg(c_frame, self.top_w, self.top_h) # (480, 848, 3)
            self.RGB_signal.emit(top_img)

            # core loop
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit)
            loop.exec_()

    def destroy_vid(self):
        self.is_running = False

class FrameArea(QtWidgets.QWidget):
    def __init__(self, w, h, parent=None):
        super(FrameArea, self).__init__(parent)
        self.image = QtGui.QImage()
        self.w = w
        self.h = h
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setFixedSize(self.w, self.h)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
    
    @QtCore.pyqtSlot(QtGui.QImage)
    def setFrame(self, image):
        if image.isNull():
            sys.stdout.write("Frame error!\n")
            sys.stdout.flush()

        self.image = image
        self.setFixedSize(self.w, self.h)
        self.update()

class CURDDemoApp(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.is_1st_init_done = False

        # vid
        self.thread = QtCore.QThread()       
        self.vid = VideoObj()
        self.vid.complete_signal.connect(self.end_detect)
        self.vid.moveToThread(self.thread)
        self.thread.start()
    
        # status and timer
        self.status = 0

        self.setWindowTitle("Face Recognition Demo")
        self.setWindowIcon(QtGui.QIcon("imgs/icon_CURD.ico"))
        self.setStyleSheet(CURD_style['label'])

        # Button ==================
        self.btn1 = QtWidgets.QPushButton('Start')
        self.btn1.setStyleSheet(CURD_style['btn1_s_abled'])
        self.btn1.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        
        self.btn3 = QtWidgets.QPushButton('Gallery')
        self.set_btn_disable(self.btn3)
        self.btn3.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.btn_close = QtWidgets.QPushButton('Close')
        self.btn_close.setStyleSheet(CURD_style['btn4_s_abled'])
        self.btn_close.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # Button callback =========
        self.btn1.clicked.connect(self.btnStart_clicked)      
        self.btn_close.clicked.connect(self.quit_instance)

        self.btn1.setFixedSize(160,40)
        self.btn3.setFixedSize(160,40)
        self.btn_close.setFixedSize(160,40)

        # self.btn3.clicked.connect(self.reset_btn_clicked)

        self.initUI()

    def initUI(self):
        bgImage = QtGui.QImage("./imgs/bg.png")
        palette = QtGui.QPalette()
        palette.setBrush(10, QtGui.QBrush(bgImage))
        self.setPalette(palette)

        frame_view1 = FrameArea(896, 504)
        self.vid.RGB_signal.connect(frame_view1.setFrame)

        # Layout ==========================================
        main_Layout = QtWidgets.QVBoxLayout()
        top_layout = QtWidgets.QVBoxLayout()
        btn_layout = QtWidgets.QHBoxLayout()

        main_Layout.setContentsMargins(50, 86, 50, 0)

        # top
        top_layout.addWidget(frame_view1, alignment=Qt.AlignHCenter)
        top_layout.setContentsMargins(0, 5, 0, 0)
        main_Layout.addLayout(top_layout)

        btn_layout.addWidget(self.btn1, alignment=Qt.AlignTop) # , alignment=Qt.AlignVCenter
        btn_layout.addWidget(self.btn3, alignment=Qt.AlignTop)
        btn_layout.addWidget(self.btn_close, alignment=Qt.AlignTop)
        btn_layout.setContentsMargins(0, 13, 0, 0)

        main_Layout.addLayout(btn_layout)

        self.setLayout(main_Layout)
        self.setFixedSize(1024,708)
        self.move(500,20)
        self.show()

    def init_sdk(self):
        result = self.vid.sdk_init()
        if not result:
            QtWidgets.QMessageBox.warning(self, "INIT ERROR", "Plz Check Camera, pth file or other resources !\nThen Restart this program.",\
                QtWidgets.QMessageBox.Ok)
            self.quit_instance()
        else:
            return True

    def set_btn_disable(self, btn):
        btn.setDisabled(True)
        btn.setStyleSheet(CURD_style['btn_disabled'])

    def set_btn_able(self, btn, btn_name):
        btn.setEnabled(True)
        btn.setStyleSheet( CURD_style.get(btn_name) )

    def btnStart_clicked(self):
        if not self.vid.is_sdk_init_done():
            self.set_btn_disable(self.btn1)
            sys.stdout.write("\tLoading... please wait some time..\n")
            sys.stdout.flush()
            result = self.init_sdk()
            if not result:
                self.quit_instance()
            else:
                self.vid.changeStatus(1) # idle

        # btn abled status
        if not self.vid.is_running:
            self.vid.is_running = True
            self.btn1.setText('Running')
            self.set_btn_disable(self.btn1)

            vid_result = self.vid.start_cam()
            if not vid_result:
                self.quit_instance()

    @QtCore.pyqtSlot()
    def end_detect(self):
        self.vid.changeStatus(30)
        if self.detecting_timer.isActive():
            self.detecting_timer.stop()
    
    def quit_instance(self):
        self.vid.destroy_vid()
        sys.exit()
        QCoreApplication.instance().quit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    sys.stdout.write("\n*****************************************\n")
    sys.stdout.write("***** CUBOX Face Recognition Program\n")
    sys.stdout.write("*****************************************\n\n")
    sys.stdout.write("==> Click the [Start] button to start this program\n")
    sys.stdout.flush()
    CURD_demo = CURDDemoApp()
    sys.exit(app.exec_())