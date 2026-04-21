1. Vào trang release Poppler for Windows:
https://github.com/oschwartz10612/poppler-windows/releases
2. Tải file zip bản mới nhất, ví dụ:
Release-xx.x.x-0.zip
3. Giải nén vào thư mục bạn muốn, ví dụ:
D:\poppler
4. Thư mục cần add PATH là:
D:\poppler\Library\bin
5. Add PATH tạm thời trong terminal hiện tại (PowerShell):
$env:Path += ";D:\poppler\Library\bin"

6. Kiểm tra:
pdftoppm -h
hoặc
pdfinfo -h

Nếu hiện help là cài xong.

7. Add PATH vĩnh viễn:
Mở Start, tìm “Environment Variables”
Chọn Edit the system environment variables
Environment Variables
Chọn Path (User hoặc System) -> Edit -> New
Dán D:\poppler\Library\bin
OK hết và mở terminal mới để nhận PATH.

8. Bat/tat OCR fallback trong du an:
Mo file .env va doi gia tri OCR_FALLBACK_ENABLED=true hoac false.