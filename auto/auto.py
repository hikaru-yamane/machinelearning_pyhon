# coding: utf-8
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import webbrowser
import requests
import bs4
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pyautogui
import pyperclip # クリップボードにコピー
import subprocess # アプリ起動
import ctypes # クリック検出
from collections import OrderedDict


## ファイル管理
## shutilモジュールでファイル・フォルダのコピー・移動・削除
## フォルダ作成
#os.makedirs('C:\\Users\\yamane\\Desktop\\aa')
##os.makedirs(r'C:\Users\yamane\Desktop\aa') # エスケープ文字無視
## ファイルサイズ
#path = 'C:\\Users\\yamane\\Desktop\\nn_python'
#total_size = 0
#max_size = 0
#for file_name in os.listdir(path):
#    file_size = os.path.getsize(os.path.join(path, file_name))
#    total_size += file_size
#    if file_size > max_size:
#        max_size = file_size
#        max_file = file_name
#print('max_file: {}'.format(max_file))


## 時間制御
## datatimeで年月日
## threading.Threadで複数スレッド(やめて)
## Popen()でほかのpythonプログラムを実行可能(やめて)
#start_time = time.time()
## sleep(3)だとCtrl+Cで途中で止められない
#for i in range(3):
#    time.sleep(1) # 3秒待つ
#end_time = time.time()
#measured_time = end_time - start_time
#print('計測時間: {}s'.format(measured_time))


## メール
## 安全性の低いアプリからのアクセスを許可
## アプリ固有のパスワードが必要ならhttps://support.google.com/accounts/answer/185833
### 英語のみ
##import smtplib
##smtp_obj = smtplib.SMTP('smtp.gmail.com', 587)
##smtp_obj.ehlo()
##smtp_obj.starttls()
##smtp_obj.login('my@gmail.com', 'PASSWORD')
##smtp_obj.sendmail('my@gmail.com', 'to@gmail.com', 'Subject:aa\nhello!') # \n必須
##smtp_obj.quit()
## 日本語
#charset = 'iso-2022-jp' # 標準の文字コード
#msg = MIMEText('日本語本文', 'plain', charset)
#msg['Subject'] = Header('日本語件名'.encode(charset), charset)
#my_mail_add = 'hikaru.yamane4@gmail.com'
#to_mail_add = 'hikaru.yamane4@gmail.com' # 複数ならリスト
#password = input("password: ")
## 設定
#smtp_obj = smtplib.SMTP('smtp.gmail.com', 587) # サーバ名 ポート番号(検索すれば出る)
##smtp_obj = smtplib.SMTP_SSL('smtp.gmail.com', 465) # 上でダメなら
#smtp_obj.ehlo() # smtpに挨拶(接続に成功したか分かりこれがないとエラーになる)
#smtp_obj.starttls() # TLS暗号化(SSLは暗号化済みなので不要)
#smtp_obj.login(my_mail_add, password) # input()で入力すること
#smtp_obj.sendmail(my_mail_add, to_mail_add, msg.as_string())
#smtp_obj.quit()


## webスクレイピング
## Scrapyでクローラ
#url = 'http://imoandpotato.html.xdomain.jp/'
#webbrowser.open(url)
#res = requests.get(url) # ページのダウンロード
#res.encoding = res.apparent_encoding # 文字化け防止
##print(res.text[-100:])
#soup = bs4.BeautifulSoup(res.content, 'html.parser')
#elems = soup.select('title')
#print(elems[0].getText())
### 検索結果表示→うまくいかない
##keyword = '山根光のホームページ'
##url = 'https://www.google.com/search?q=' + keyword
##link_elems = soup.select('.g a') # class='g'の<link>要素
##num_open = min(2, len(link_elems)) # 最大で2つタブを開く
##for i in range(num_open):
##    webbrowser.open(link_elems[i].get('href'))

# selenium
# find_element_*:最初の１つ， find_elements_*:すべて
# なぜかwebdriverのpathが通らないから直接記述
url = 'https://cloudlatex.io/users/sign_in'
browser = webdriver.Chrome(executable_path=r'C:/chromedriver_win32/chromedriver.exe') # ブラウザ立ち上げ
browser.get(url) # ページ開く
browser.maximize_window()
# 対象がないとエラーが出て止まってしまうから回避
try:
    #link_elem = browser.find_element_by_link_text('自己紹介')
    #link_elem.click()
    email_elem = browser.find_element_by_id('user_name_or_email')
    email_elem.send_keys('mail')
    password_elem = browser.find_element_by_id('user_password')
    password_elem.send_keys('password')
    password_elem.submit()
except:
    print('なし')
## ブラウザボタン
#browser.back() # 戻る
## 特殊キー送信
#html_elem = browser.find_element_by_tag_name('html')
#html_elem.send_keys(Keys.END) # 末尾にスクロール # スクロールしてロードされるサイトに有効


## gui
## ドキュメント http://pyautogui.readthedocs.org/
## 座標：左上原点で→x，↓y (正の整数)
## Ctrl+Alt+Deleteでログアウト(強制停止)
## ポインタを左上に動かせば止められる →pyautogui.FAILSAFE = False でこの機能無効
## Ctrl+C: コンソールがアクティブなら有効
## フォーム入力はTabで移動すれば座標を調べる回数を減らせる
## 自動化が間違って動作し続けたら大変．色認識や画像認識を用いて異常終了させよう
## import mouseでイベントを取得，座標を保存すればお絵かき可能
#time.sleep(2) # waitkey
#width, height = pyautogui.size()
#pyautogui.PAUSE = 1 # gui操作のたびに休止(デバッグで使用) # デフォルトで0.1→0にすると止められない？
#pyautogui.moveTo(100, 100, duration=0.25) # 絶対座標 # duration=0がデフォルト
#pyautogui.moveRel(-2, 0, duration=0.5) # 相対座標
#print(pyautogui.position()) # 現在の位置
#pyautogui.click(100,150,button='right')
#pyautogui.doubleClick()
#pyautogui.dragTo()
#pyautogui.dragRel(0,100,duration=1) # 相対的な移動量
#pyautogui.scroll(100,x,y) # ポインタのあるwindowをスクロール
## スクリーンショット
#img = pyautogui.screenshot()
#img.getpixel((450,720)) # (240,240,240) 座標のRGBを返す
## 色認識
#pyautogui.pixelMatchesColor(450,720,(240,240,240)) # 座標の色と引数のRGBが一致してればTrue
## 画像認識
## snipping toolでpng
#img_pos = pyautogui.locateOnScreen('auto_image.png') # (10,100,23,33) 画像の座標を返す 左上点座標x,y,幅,高さ
#img_center = pyautogui.center(img_pos) # (50,80) 中心点
#pyautogui.click(img_center)
#list(pyautogui.locateAllOnScreen('submit.png')) # 複数ならリスト
## 文字列送信
#pyautogui.click() # アクティブにする
#pyautogui.typewrite('hello')
#pyautogui.typewrite('hello', 0.25) # ゆっくり１文字ずつ送信．遅いアプリに有効
## typewriteは日本語未対応→クリップボードを経由 # 全角でローマ字入力もある
#pyperclip.copy('あああ') # クリップボードにコピー
#pyautogui.hotkey('ctrl', 'v') # 貼り付け
## キー送信
#pyautogui.typewrite('a','left','B') # 'Ba': a,左矢印,B
## キーボードの押下・解放
#pyautogui.keyDown('shift')
#pyautogui.press('4') # 押下・解放
#pyautogui.keyUp('shift')
## ホットキー
#pyautogui.hotkey('ctrl', 'c')
## 現在の座標表示
#print('中断: Ctrl+C')
#try:
#    while True:
#        time.sleep(1)
#        print(pyautogui.position())
#except KeyboardInterrupt:
#    print('\n終了')


## お絵かき
## 実行時に変数を削除しないよう設定
## 入力
##log_pos = OrderedDict()
##leftbutton = 0x01 # マウス左ボタン
##flag_down = 0
##cnt = 0 # キーを重複させないために付与
##try:
##    while True:
##        state_down = ctypes.windll.user32.GetAsyncKeyState(leftbutton)
##        if state_down > 0 and flag_down == 1:
##            log_pos['move'+str(cnt)] = pyautogui.position()
##            cnt += 1
##        elif state_down > 0 and flag_down == 0: # ２回評価しないよう２番目にした
##            log_pos['down'+str(cnt)] = pyautogui.position()
##            flag_down = 1
##            cnt += 1
##        elif state_down == 0 and flag_down == 1:
##            log_pos['up'+str(cnt)] = pyautogui.position()
##            flag_down = 0
##            cnt += 1
##        time.sleep(0.01)
##except KeyboardInterrupt:
##    print('\n終了')
## 出力
#pyautogui.PAUSE = 0.01 # デフォルト0.1は遅すぎる
##pyautogui.hotkey('win', 'down') # 最小化
##pyautogui.hotkey('win', 'down') 
##subprocess.Popen(r'C:\WINDOWS\system32\mspaint.exe') # 起動
##pyautogui.hotkey('win', 'up')
#for key in log_pos.keys():
#    if 'move' in key:
#        pyautogui.moveTo(log_pos[key][0], log_pos[key][1], duration=0)
#    elif 'down' in key:
#        pyautogui.mouseDown(button='left', x=log_pos[key][0], y=log_pos[key][1])
#    elif 'up' in key:
#        pyautogui.mouseUp(button='left', x=log_pos[key][0], y=log_pos[key][1])

