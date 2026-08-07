# V7 Design Repair

V7 dành cho poster/catalogue raster vừa mờ vừa có chữ gãy, thiếu dấu hoặc sai
chính tả. Ảnh và model đều chạy cục bộ. V7 không đổi đuôi giả và không gọi một
ảnh raster là vector.

## Đường xử lý mặc định

1. Chuẩn hóa ảnh nhưng vẫn dùng ảnh nguồn cho OCR và hình học chữ.
2. PP-OCRv6 và Tesseract tiếng Việt cung cấp bằng chứng đọc; nội dung chưa chắc
   chắn phải được người dùng duyệt.
3. Chỉ chữ đã duyệt mới được gỡ. Nền bị chữ che là phần tổng hợp có giới hạn,
   không được mô tả là pixel gốc đã khôi phục.
4. Toàn bộ clean raster đi qua các bước classical có gate: deblur luminance có
   giới hạn, denoise theo mức nhiễu và tăng chi tiết quan sát được. Từng bước tự
   rollback nếu không đạt.
5. V7 gọi một checkpoint `Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth` qua V3
   PyTorch/CUDA ở chế độ single. Đường V7 này không dùng GAN, HAT, Real-ESRGAN
   hoặc fusion ba model.
6. Swin2SR luôn tạo dự đoán native x4. Với `n=1`, kết quả x4 được downsample có
   kiểm soát về x1; với `n=2..20`, kết quả được resample một lần tới đúng xN.
   Chữ Unicode đã duyệt được vẽ lại sau đó ở kích thước cuối.

Super-resolution dự đoán pixel hợp lý từ dữ liệu nhìn thấy; nó không phục hồi
được bằng chứng đã mất và không bảo đảm nội dung nhỏ vốn không đọc được là đúng.
Nếu backend lỗi hoặc ứng viên SR không qua gate, V7 dùng fallback classical/
Lanczos. Trạng thái, lý do và việc model có thật sự được chấp nhận hay không phải
nằm trong `_KY_THUAT/QA.json` và `_KY_THUAT/manifest.json`; bundle buộc ở trạng
thái cần duyệt thay vì tự nhận là kết quả đã đạt.

## Hợp đồng an toàn

- PP-OCRv6 và Tesseract là bằng chứng, không phải sự thật.
- Các augmentation PP-OCRv6 vẫn chỉ là một engine. Tesseract chỉ bỏ phiếu khi
  có `vie.traineddata` đã kiểm tra; không gắn nhãn tiếng Việt cho fallback English.
- Giá, số điện thoại, địa chỉ, SKU, chữ số và tên thương hiệu không được âm thầm
  sửa. Đề xuất của model ngôn ngữ luôn phải được duyệt.
- Review được khóa bằng SHA-256 nguồn và fingerprint gồm bbox + OCR NFC. Hồ sơ
  thiếu/sai/cũ bị từ chối an toàn.
- V7 đo tight old-ink mask, định hình bằng RAQM/HarfBuzz, kiểm tra glyph tiếng
  Việt và chỉ dùng các trục OpenType `wght`/`wdth` được font hỗ trợ.
- Geometry, tâm, clipping, ghost, seam và khóa typography từ source tới xN đều
  là gate. OCR đọc ngược chỉ là cảnh báo vì OCR có thể đọc sai dấu nhỏ đã render đúng.
- Bundle chưa duyệt hoặc lỗi gate mang `REVIEW_REQUIRED`/`FAILED_QA`; không được
  gọi là file giao in.
- SVG có chữ Unicode chỉnh sửa thật nhưng phần hình/nền vẫn là raster và font
  Windows không được nhúng.

## Cài và chạy

`V7/.venv` chứa Paddle CPU riêng để tránh xung đột DLL/NumPy/oneDNN. Phần phục
dựng raster mặc định dùng runtime V3 PyTorch/CUDA riêng. Vì Swin2SR chạy cả ở
`n=1`, V7 mặc định cần GPU NVIDIA và checkpoint V3 đã cài.

```powershell
cd "C:\Users\Admin\Desktop\RESIZE"
powershell -ExecutionPolicy Bypass -File .\APP\engines\V7\setup_v7.ps1

.\upscale repair poster.png 1
.\upscale repair poster.png 4
.\upscale repair poster.png 4 --review defer
```

Với một ảnh, chế độ mặc định mở trang duyệt cục bộ trên `127.0.0.1` và ưu tiên
Chrome. Muốn mở lại màn hình duyệt của một bundle:

```powershell
.\upscale review "C:\duong-dan\poster_V7_REPAIR_x1"
# Bí danh tiếng Việt:
.\upscale duyet "C:\duong-dan\poster_V7_REPAIR_x1"
```

Người dùng bấm **Lưu và hoàn tất** trên trang rồi chạy lại lệnh `repair` được chương trình
in ra. Mỗi quyết định vùng được lưu ngay khi bấm; không cần mở `TEXT_REVIEW.json` bằng tay.

Nếu chỉ cần làm rõ raster mà không sửa nội dung, bấm **Giữ nguyên tất cả vùng còn lại**. Đây là một
bulk-keep nguyên tử: nó giữ bitmap, không áp dụng đề xuất OCR và không thay chữ, giá hay mã hàng.
Sau đó bấm **Lưu và hoàn tất** để đóng phiên; không cần nhập lại chữ.

Với thư mục, chạy lượt đầu ở `defer`, duyệt từng bundle, rồi chạy lại đúng lệnh
thư mục. Launcher tự nạp hồ sơ đúng của từng ảnh:

```powershell
.\upscale repair "D:\BO ANH" 1 --review defer
.\upscale review "D:\duong-dan\mot_bundle_V7"
.\upscale repair "D:\BO ANH" 1 --review defer
```

## Đầu ra dành cho người dùng

```text
<ten>_V7_REPAIR_xN\
  01_KET_QUA_DA_DAT.png          chỉ khi PASS
  # hoặc 01_XEM_TRUOC_CAN_DUYET.png khi chưa PASS
  02_SO_SANH.png
  03_CHINH_SUA.svg
  _KY_THUAT\
    SOURCE.png
    CLEAN.png
    OVERLAY.png
    TEXT_REVIEW.json
    QA.json
    manifest.json
```

Ba file đánh số là giao diện file đơn giản. `_KY_THUAT` chứa bằng chứng kỹ thuật;
giao diện review tự đọc/ghi JSON trong đó. Trước khi dùng ảnh quan trọng, mở
`02_SO_SANH.png` ở 100%, đọc lại toàn bộ chữ/số và kiểm tra trong QA rằng
`raster_restore.super_resolution.accepted` là `true`. Nếu là `false`, kết quả đang dùng fallback
được khai báo chứ không phải dự đoán Swin2SR đã được chấp nhận.
