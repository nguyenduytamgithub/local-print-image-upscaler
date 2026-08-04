# V6 — Ý tưởng phục dựng ảnh chất lượng thấp

**Trạng thái:** mới ghi nhận ý tưởng để tiếp tục bàn bạc. Chưa viết code V6, chưa cài mô hình V6,
chưa thay đổi V2/V3/V4/V5. V7 Design Repair hiện là nhánh triển khai riêng cho bài toán sửa chữ/nền
poster; nó không thay thế phạm vi phục hồi ảnh chụp rộng hơn dự kiến cho V6.

## Bài toán V6

Đầu vào không chỉ có độ phân giải thấp mà còn có thể bị mờ nét, rung/chuyển động, nhiễu, nén JPEG, chói sáng,
ánh sáng không đều, sai phối cảnh và mất chi tiết. Mục tiêu của V6 là **phục dựng nội dung trước**, sau đó mới phóng lớn
và chuẩn bị file in/chỉnh sửa.

V6 không được hiểu là “nhân thêm pixel” hoặc chỉ đổi PNG sang PDF/SVG. Nó là một quy trình phục dựng khác với
V2/V3/V4/V5.

## Hai loại đầu ra dự kiến

### 1. RESTORED_PHOTO — ảnh chụp đã phục hồi

- Nắn phối cảnh mặt phẳng khi ảnh là bảng hiệu, poster hoặc tài liệu chụp xiên.
- Sửa ánh sáng không đều, ám màu và vùng chói trong giới hạn còn dữ liệu.
- Khử nhiễu, giảm artefact JPEG, giảm mờ/rung.
- Chạy super-resolution có kiểm soát và chia tile để vừa RTX 3060 12 GB.
- Giữ hình ảnh tự nhiên; không tự ý thay chữ, số điện thoại, giá hoặc logo.
- Xuất PNG/TIFF chất lượng cao cùng báo cáo những vùng không thể khôi phục chắc chắn.

### 2. EDITABLE_REBUILD — dựng lại thiết kế để sửa và in

Đây là hướng phù hợp nhất với bảng hiệu/poster có chữ và mảng đồ họa rõ:

- Nắn phẳng bảng hiệu bằng hình học/phối cảnh.
- OCR tiếng Việt để đọc chữ, nhưng bắt buộc cho người dùng kiểm tra nội dung quan trọng.
- Dựng lại chữ thành text/vector thật thay vì phóng lớn pixel chữ bị mờ.
- Vector hóa logo, khung, đường trang trí và các mảng màu phù hợp.
- Tái tạo nền sạch; phần bị vật thể che phải ghi rõ là nội dung suy đoán.
- Xuất SVG/PSD hoặc OpenRaster để chỉnh sửa, PDF/X-4 để in và PNG để xem nhanh.

## Pipeline dự kiến

1. Phân tích ảnh và phân loại: ảnh tự nhiên, bảng hiệu/poster, tài liệu hoặc ảnh hỗn hợp.
2. Phát hiện mặt phẳng và nắn phối cảnh bằng OpenCV/homography.
3. Phục hồi ánh sáng, màu, nhiễu và độ mờ bằng mô hình chạy cục bộ.
4. Với bảng hiệu/poster: phát hiện vùng chữ, OCR tiếng Việt, xác nhận lại chữ quan trọng.
5. Tách nội dung thành text, vector, ảnh và nền; dựng lại thiết kế có thể chỉnh sửa.
6. Phóng lớn raster còn lại bằng engine chất lượng cao; ưu tiên tái sử dụng V3.
7. Đóng gói đầu ra in/chỉnh sửa bằng các phần đã kiểm chứng từ V4 và V5.
8. Chạy QA toàn ảnh: chữ, màu, biên, ringing, artefact tile, kích thước và độ phân giải in.

## Thành phần mã nguồn mở đáng nghiên cứu khi bắt đầu code

- OpenCV: nắn phối cảnh và biến đổi hình học xác định.
- Restormer hoặc NAFNet: khử mờ, khử nhiễu và phục hồi ảnh.
- DocRes: tham khảo pipeline phục hồi tài liệu nhiều tác vụ; chỉ dùng khi phù hợp với ảnh màu/bảng hiệu.
- PaddleOCR: OCR tiếng Việt và tọa độ chữ.
- LayerD: tham khảo phân rã thiết kế raster thành layer SVG/PSD.
- VTracer/LIVE: vector hóa logo và mảng đồ họa phẳng.
- LaMa hoặc phương pháp tương đương: điền nền bị che, nhưng luôn đánh dấu là nền tổng hợp.
- V3/V4/V5 hiện tại: tái sử dụng phần upscale, PDF/X-4/SVG và layer thay vì viết lại từ đầu.

## Nguyên tắc chất lượng và trung thực

- Không tuyên bố khôi phục “chi tiết gốc” khi dữ liệu đã mất; AI chỉ có thể tạo chi tiết hợp lý.
- Chữ, số điện thoại, địa chỉ, giá, mã sản phẩm và logo phải được xác minh trước khi in.
- Luôn giữ ảnh nguồn và tạo đầu ra mới; không ghi đè.
- So sánh có đối chứng với ảnh nguồn và V3, không chọn kết quả chỉ vì nhìn sắc hơn.
- Đánh giá toàn ảnh, không làm nét chắp vá vài vùng.
- Có chế độ trung thực và chế độ dựng lại; không trộn hai khái niệm mà không ghi rõ.
- V6 phải chạy tuần tự/chia tile trên RTX 3060 12 GB; cloud chỉ là tùy chọn, không phải điều kiện bắt buộc.

## Tính khả thi đã ghi nhận

- Phục hồi ảnh chụp nói chung: khả thi để cải thiện rõ, nhưng không bảo đảm lấy lại đúng chi tiết đã mất.
- Dựng lại bảng hiệu/poster đơn giản: khả thi cao vì chữ, hình học, logo và mảng màu có thể tái tạo thành đối tượng thật.
- Một mô hình tự động hoàn toàn, vừa nhỏ vừa bảo đảm đúng mọi chữ và mọi layer: hiện chưa nên xem là giải pháp tin cậy.
- Hướng thực tế nhất là pipeline lai: hình học xác định + mô hình phục hồi + OCR + vector/layer + bước duyệt của người dùng.

## Những quyết định sẽ bàn sau, trước khi code

- V6 ưu tiên ảnh tự nhiên hay bảng hiệu/poster trước.
- Mức tự động và màn hình/bước xác nhận OCR.
- Định dạng layer chính: PSD, OpenRaster, SVG hay kết hợp.
- Model nào cho chất lượng tốt nhất trong giới hạn RTX 3060 12 GB và giấy phép sử dụng.
- Bộ ảnh kiểm thử, tiêu chí đạt và kích thước in mục tiêu.
- Cú pháp lệnh, cấu trúc INPUT/OUTPUT và cách đóng gói cài đặt.

## Nguồn nền tảng để xem lại khi triển khai

- OpenCV geometric transformations: https://docs.opencv.org/4.13.0/da/d6e/tutorial_py_geometric_transformations.html
- Restormer: https://github.com/swz30/Restormer
- NAFNet: https://github.com/megvii-research/NAFNet
- DocRes: https://github.com/ZZZHANG-jx/DocRes
- PaddleOCR: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html
- LayerD: https://github.com/CyberAgentAILab/LayerD
- LIVE layer-wise vectorization: https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization
- VTracer: https://github.com/visioncortex/vtracer
- LaMa: https://github.com/advimman/lama
- Perception–distortion tradeoff: https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html
