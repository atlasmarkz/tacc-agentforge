@echo off
"C:\Program Files\Git\bin\bash.exe" "%~dp0rsync-wrap.sh" %*
exit /b %errorlevel%
