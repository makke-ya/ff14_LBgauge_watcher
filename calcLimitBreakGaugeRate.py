# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import time
import threading
import wave
import pyaudio
import datetime
import signal
from PIL import ImageGrab
from PIL import Image
import win32api
import win32gui
import win32ui
import win32con

wav_margin = 3.0
#maxgauge_th = 0.9
init_gauge_th = 0.95
gauge_th = 0.95
rightgauge_th = 0.7
loop_interval = 0.5
width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
imgrab_lt = (left, top)
imgrab_wh = (width, height)
print(imgrab_lt, imgrab_wh)

class CalcLimitBreakGaugeRate():
    def __init__(self):
        self.wf_1 = "wav/LBgauge1.wav"
        self.wf_2 = "wav/LBgauge2.wav"
        self.wf_90 = "wav/LBgauge90%.wav"
        self.wf_max = "wav/LBgaugeMAX.wav"
        self.mark_lb = cv2.imread("mark/limitbreak.tif")
        self.mask_lb = cv2.imread("mark/limitbreak_mask.tif")
        #self.mark_max = cv2.imread("mark/maxgauge.tif")
        self.mark_gauge = cv2.imread("mark/nonegauge.tif")
        self.mask_gauge = cv2.imread("mark/gauge_mask.tif")
        #self.mark_right = cv2.imread("mark/nonegauge_right.tif")
        self.end_flag = False
        signal.signal(signal.SIGINT, self.handler)
        self.gauge1_flag = False
        self.gauge2_flag = False
        self.ths = []
        self.gauge_rate = []
        self.before_wav = datetime.datetime.now()

    def handler(self, signal, frame):
        self.end_flag = True
        for t in self.ths:
            t.join()

    def playAudio(self, wf_name):
        self.wf = wave.open(wf_name, "rb")
        p = pyaudio.PyAudio() # PyAudioのインスタンスを生成 (1)
        
        # Streamを生成(3)
        stream = p.open(format=p.get_format_from_width(self.wf.getsampwidth()),
                        channels=self.wf.getnchannels(),
                        rate=self.wf.getframerate(),
                        output=True,
                        stream_callback=self.callback)
        
        # Streamをつかって再生開始 (4)
        stream.start_stream()
        
        # 再生中はひとまず待っておきます (5)
        while stream.is_active():
            time.sleep(0.1)
        
        # 再生が終わると、ストリームを停止・解放 (6)
        stream.stop_stream()
        stream.close()
        self.wf.close()
        
        # close PyAudio (7)
        p.terminate()
    
    # 再生用のコールバック関数を定義
    def callback(self, in_data, frame_count, time_info, status):
        data = self.wf.readframes(frame_count)
        return (data, pyaudio.paContinue)
    
    def screenshot(self, x, y, w, h):
        """ スクリーンショット撮ってそれを(Pillow.Imageで)返す """
        window = win32gui.GetDesktopWindow()
        window_dev = win32gui.GetWindowDC(window)
        window_dc = win32ui.CreateDCFromHandle(window_dev)
        compatible_dc = window_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(window_dc, w, h)
        compatible_dc.SelectObject(bmp)
        compatible_dc.BitBlt((0, 0), (w, h), window_dc, (x, y), win32con.SRCCOPY)
        img = Image.frombuffer('RGB', (w, h), bmp.GetBitmapBits(True), 'raw', 'BGRX', 0, 1)
        win32gui.DeleteDC(compatible_dc.GetHandleAttrib())
        win32gui.DeleteObject(bmp.GetHandle())
        win32gui.ReleaseDC(win32gui.GetDesktopWindow(), window_dev) 
        return img
    
    def calcGaugeRate(self, im):
        gauge_rate = []
    
        # ゲージの左と右のポイントを探す
        gauge_pts = self.searchMatchTempPts(self.mark_gauge, im, gauge_th,
                self.mask_gauge)
        gauge_pts = sorted(gauge_pts, key = lambda x: x[0])
        #print("gauge", gauge_pts)
        
        # ゲージのx軸のみ取り出し
        lb_pts = []
        for p in gauge_pts:
            y = p[1] + 6
            lb_pts.append([(p[0] + 17, y), (p[0] + 143, y)])
        
        #print("lb", lb_pts)
        gauge_num = len(lb_pts)
        if gauge_num > 3:
            print("ERROR: gauge > 3")
            return gauge_rate
        #print(gauge_num)
    
        # ゲージの割合を確認
        for l, r in lb_pts:
            width = r[0] - l[0]
            if width < 0:
                return gauge_rate
            nonegauge = 0
            y = l[1]
            for x in range(l[0], r[0]):
                val = im[y, x]
                if val[0] < 200 and val[2] < 200:
                    im[y-3:y+3, x] = [0, 0, 255]
                    nonegauge += 1
            gauge_rate.append((1.0 - nonegauge / width) * 100.0)
        if gauge_num > 0:
            cv2.imwrite("lb_im_red.tif", im)
        return gauge_rate
    
    def getNearPtIdx(self, com_pt, pts, margin=10):
        ret = []
        for i, pt in enumerate(pts):
            diff = (com_pt[0] - pt[0], com_pt[1] - pt[1])
            if abs(diff[0]) < margin and abs(diff[1]) < margin:
                ret.append(i)
        return ret
    
    def searchMatchTempPts(self, mark, im, th=0.9, mask=None):
        #res = cv2.matchTemplate(im, mark, cv2.TM_CCOEFF_NORMED)
        pts = []
        res = cv2.matchTemplate(im, mark, cv2.TM_CCORR_NORMED, None, mask)
        if np.isnan(np.sum(res)):
            res = res[~np.isnan(res)]
            print("                                    ---", end="\r")
        loc = np.where(res >= th)
        if len(loc[0]) > 0:
            pts = [pt for pt in zip(*loc[::-1])]
            #scores = [res[pt[1], pt[0]] for pt in zip(*loc[::-1])]
            #print(scores)
            pts = sorted(pts, key=lambda x: res[x[1], x[0]], reverse=True)
    
            for i in range(3):
                del_idx = self.getNearPtIdx(pts[i], pts[i + 1:], margin=50)
                for idx in reversed(del_idx):
                    del pts[i + 1 + idx]
                if len(pts) < i + 2:
                    break
        return pts
        
    def mainLoop(self, set_location):
        if set_location is True:
            # テンプレートマッチング
            maxval = 0
            while(self.end_flag == False):
                # スクリーンショット読み込み(全体)
                #im = ImageGrab.grab()
                im = self.screenshot(imgrab_lt[0], imgrab_lt[1],
                                     imgrab_wh[0], imgrab_wh[1])
                im = np.asarray(im)                    
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                cv2.imwrite("im.tif", im)
                im = cv2.imread("im.tif")

                res = cv2.matchTemplate(im, self.mark_lb, cv2.TM_CCORR_NORMED,
                        None, self.mask_lb)
                (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(res)
                print("({0}, {1}) score = {2}".format( \
                        maxloc[0], maxloc[1], maxval))
                if maxval >= init_gauge_th:
                    break
                print("Nothing LB gauge")
                time.sleep(5)

            # リミットブレイクゲージの画像を切り取り
            lt = (imgrab_lt[0] + maxloc[0] - 15, imgrab_lt[1] + maxloc[1])
            wh = (500, 25)
            rb = wh
            #rb = (maxloc[0] + 500, maxloc[1] + 25)
            #lt = (1274, 513)
            #rb = (1789, 538)
        else:
            lt = (1274, 513)
            rb = (1789, 538)
        print(lt, rb)
    
        while(self.end_flag == False):
            try:
                #lb_im = ImageGrab.grab((lt[0], lt[1], rb[0], rb[1]))
                lb_im = self.screenshot(lt[0], lt[1], wh[0], wh[1])
                lb_im = np.asarray(lb_im)                    
                lb_im = cv2.cvtColor(lb_im, cv2.COLOR_BGR2RGB)
                #lb_im = im[lt[1]:rb[1], lt[0]:rb[0]]
                cv2.imwrite("lb_im.tif", lb_im)
            except:
                print("screenshot failed")
                print(win32api.FormatMessage())
                sys.exit()
    
            self.gauge_rate = self.calcGaugeRate(lb_im)
            print("[gauge_rate [{}]]".format(len(self.gauge_rate)), end="")
            for i, rate in enumerate(self.gauge_rate, start=1):
                print(" {0}:{1:3.1f}".format(i, rate), end="")
            print("                                       ", end="\r")
    
            self.now = datetime.datetime.now()
            diff = self.now - self.before_wav
            self.audio_contoroler(diff)

    def audio_contoroler(self, diff):
        if len(self.gauge_rate) < 1 or diff.total_seconds() < wav_margin:
            time.sleep(loop_interval)
            return

        # flag on/off
        if len(self.gauge_rate) > 1 and self.gauge_rate[1] == 0.0:
            self.gauge1_flag = False
        if len(self.gauge_rate) > 2 and self.gauge_rate[2] == 0.0:
            self.gauge2_flag = False

        # audio
        if self.gauge_rate[-1] == 100.0:
            self.ths.append(
                    threading.Thread(target=self.playAudio, args=(self.wf_max,)))
        elif self.gauge_rate[-1] > 90.0:
            self.ths.append(threading.Thread(
                target=self.playAudio, args=(self.wf_90,)))
        elif self.gauge1_flag == False and self.gauge_rate[0] == 100.0:
            self.ths.append(threading.Thread(
                target=self.playAudio, args=(self.wf_1,)))
            self.gauge1_flag = True
        elif self.gauge2_flag == False and len(self.gauge_rate) > 1 and \
                self.gauge_rate[1] == 100.0:
            self.ths.append(threading.Thread(
                target=self.playAudio, args=(self.wf_2,)))
            self.gauge2_flag = True
        else:
            time.sleep(loop_interval)
            return

        # wav play start
        self.ths[-1].start()
        self.before_wav = self.now

if __name__ == "__main__":
    if len(sys.argv) > 1:
        set_location = False
    else:
        set_location = True
    calc_lbr = CalcLimitBreakGaugeRate()
    calc_lbr.mainLoop(set_location)

