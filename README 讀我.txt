# 出錯時的處理

## 無效，參考
netstat -ant |find /C `"5000`"

tasklist |find "<process number>"

## 實際有用

cmd:

netstat -ano | findstr 127.0.0.1:5000

tasklist | findstr 7152

打開工作管理員確認 PID

## 如果有錯，就直接換個 PORT 吧