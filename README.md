# Local Print Image Upscaler — V2/V3/V4/V5/V7

Chương trình xử lý ảnh cục bộ trên Windows cho bảng hiệu, poster và file in khổ lớn.
Ảnh nguồn không được gửi lên dịch vụ đám mây.

## Sáu chế độ

| Chế độ | Đầu ra chính | Khi nên dùng |
|---|---|---|
| **V2 Fast** | PNG phóng lớn | Cần kết quả nhanh, máy có Vulkan |
| **V3 High** | PNG AI chất lượng cao | Ảnh chụp hoặc ảnh nhiều texture, máy có NVIDIA CUDA |
| **V4 Print** | Raster AI/USM đã qua ablation + SVG/PDF/X-4 + PNG | Bản in nghiêm túc, cần phục hồi đồng đều toàn ảnh bằng NVIDIA CUDA |
| **V4 Vector** | SVG/PDF/X-4 toàn path | Logo, chữ, mảng màu phẳng; không dùng cho ảnh chụp/gradient |
| **V5 Smart Layers** | OpenRaster + PNG/mask ZIP; PSD khi trong giới hạn | Tách ảnh phẳng thành số lượng layer raster hữu ích để sửa trong Photoshop/Krita/Photopea/Canva |
| **V7 Design Repair** | PNG phục dựng toàn ảnh + SVG có chữ Unicode chỉnh sửa được + hồ sơ duyệt/QA | Poster/catalogue vừa mờ vừa có chữ gãy, thiếu dấu hoặc sai chính tả cần người dùng xác nhận |

V5 là engine độc lập và không xóa V2/V3/V4. Nó dùng SAM 2.1 để tìm vùng, tự phân luồng
poster/đồ họa với ảnh tự nhiên nhiều texture, rồi gộp theo panel, hàng, màu, hình học và quan hệ
cha–con trong giới hạn số layer. Grounding DINO chỉ cung cấp nhãn gợi ý; Tesseract cùng model
`tessdata_best` tiếng Việt đã pin cung cấp vùng/dòng OCR khi máy có Tesseract 5.
Riêng chữ raster được làm sạch topology O/G/N và dấu tiếng Việt trước khi ViTMatte-S đã khóa
revision ước lượng alpha mép trên GPU/CPU; model không được quyền đổi hình chữ hoặc lấp lại khoảng âm.
Nền phía dưới vật thể được tái tạo nhưng luôn ghi rõ là **synthesized**: ảnh phẳng không chứa
pixel vốn bị che, nên không thuật toán nào chứng minh được nền gốc chính xác.

V7 mặc định phục dựng **toàn bộ lớp raster** chứ không chỉ vá vài vùng chữ. Nó chạy các bước deblur/denoise/
tăng chi tiết quan sát được có gate và rollback, sau đó gọi đúng **một** checkpoint Swin2SR fidelity/PSNR
cục bộ qua runtime V3 CUDA. V7 không dùng HAT/Real-ESRGAN GAN và không trộn ba model trong đường này.
Model luôn suy đoán pixel hợp lý từ ảnh đầu vào; nó không thể khôi phục bằng chứng vốn đã mất. Nếu bước
Swin2SR lỗi hoặc không vượt gate, V7 dùng fallback classical/Lanczos và ghi rõ lý do trong QA/manifest,
không âm thầm gọi fallback là kết quả AI.

Phần chữ dùng PP-OCRv6 và Tesseract làm các nguồn đọc đối chứng, đưa thay đổi nội dung cho người dùng
duyệt, chỉ gỡ nét chữ đã được chấp thuận, dựng lại nền trong đúng footprint đó rồi vẽ lại chữ Unicode
bằng font thật ở kích thước cuối. Giá, số điện thoại, địa chỉ, mã hàng và mọi đề xuất sửa chính tả không
được âm thầm thay đổi. PNG đã sửa là bản tham chiếu vị trí raster;
SVG là tài liệu **mixed raster/vector** có đối tượng `<text>` thật để sửa, nhưng phần hình/nền vẫn là bitmap
và font Windows không được đóng gói vào file. Với headline, V7 đo footprint nét cũ, định hình bằng
RAQM/HarfBuzz và có thể tối ưu trục OpenType Variable Font `wght`/`wdth` thay vì kéo giãn bitmap.
Font/face/trục đã vượt gate ở kích thước nguồn được khóa và dùng lại ở x2..x20; lượt cuối không được
tự chọn một kiểu chữ khác. Cả hình học nguồn, hình học cuối và độ trung thực của khóa font đều là gate cứng.
V7 không tự dựng lại mọi hình minh họa, không bảo đảm khôi
phục đúng chi tiết vốn đã mất và không xuất PDF/X. Muốn giao nhà in, sửa/duyệt bằng V7 trước rồi dùng
V4 Print với ICC, bleed và khổ vật lý do nhà in yêu cầu.

V7 mặc định cần NVIDIA CUDA và bộ runtime/model V3 đã cài cho cả `n=1` lẫn `n=2..20`. Ở `n=1`, Swin2SR
vẫn dự đoán bản native x4 rồi V7 downsample có kiểm soát về đúng kích thước nguồn; vì vậy đây là phục dựng
raster thật, không phải sao chép ảnh cũ. Ở mọi hệ số, chữ đã duyệt chỉ được vẽ lại sau bước raster để giữ
Unicode và hình học sạch. Mô hình ngôn ngữ tiếng Việt cục bộ chỉ đưa ra đề xuất để duyệt, không phải nguồn
sự thật và có thể được tắt bằng `--no-language-model`.

V4 không đổi đuôi giả. Chế độ `print` tạo một tài liệu mixed vector/raster đúng bản chất:
lớp raster nhìn thấy là ứng viên **đồng nhất toàn ảnh** thắng QA và ablation ở pixel native x4.
Ứng viên có thể là master V3 AI + guarded USM, một fusion Deep thật sự thắng đối chứng, hoặc V3
fallback được ghi rõ; lớp VTracer là các đường Bezier
thật, độc lập để người thiết kế chọn lọc/chỉnh sửa nhưng có opacity `0` mặc định. Không phủ trace
lên toàn ảnh vì việc biến gradient, hạt gạo và bóng mềm thành các mảng vector rời rạc gây banding
và posterization. SVG và PDF tự khai báo cả image/path; manifest ghi mode, hash và kết quả QA.
Chế độ `vector` mới cấm hoàn toàn raster, nhưng chỉ phù hợp logo/artwork phẳng. Chữ từ bitmap
trở thành outline/path, không tự trở lại thành font có thể gõ sửa.

### V4 Print kiểm định Deep hoạt động thế nào

1. Tạo hoặc tái sử dụng master V3 native x4 đã kiểm định.
2. Chạy HAT `Real_HAT_GAN_sharper` lần nữa **đồng đều trên toàn bộ master x4**. Chia tile chỉ để
   vừa VRAM; mọi vùng đều qua cùng model, các tile chồng lấn được hòa trộn bằng trọng số cosine.
3. Chấm riêng từng crop ở đúng thang pixel native x4, có halo/guard và không ghép mosaic. Mỗi ứng viên
   phải giữ SSIM/edge, chặn ringing, đồng thời tăng ít nhất một chỉ báo **edge-strength** và một chỉ báo
   **energy/detail**; ít nhất 75% crop phải qua toàn bộ gate.
4. So guarded-USM không Deep với các fusion Deep. Deep chỉ được chọn nếu qua gate, tăng coverage tối
   thiểu hai crop và không giảm retention so với đối chứng; poster thử nghiệm này chọn trung thực V3 +
   USM vì Deep không đem lại lợi ích nhìn thấy đủ lớn.
5. Render đúng một raster xN theo stripe có halo, rồi chấm lại chính PNG đã ghi ở native x4. Nếu kết quả
   thật không lặp lại mô phỏng, V4 fallback V3. Raster được xác nhận là lớp in nhìn thấy; VTracer vẫn là
   layer ẩn để chỉnh sửa. Bundle SVG/PDF/PNG chỉ được công bố sau QA.

Vì vậy V4 không phải “làm nét vài khối”: mọi ứng viên dùng cùng một quy tắc trên toàn ảnh; Deep HAT
cũng chạy đồng đều khi được đánh giá. Vector không được dùng để vá độ nét và không được phép làm
đổi màu/gradient của lớp in mặc định.

> Super-resolution có thể phục hồi cấu trúc nhìn hợp lý nhưng không chứng minh được chi tiết vốn
> đã mất. Giá tiền, số điện thoại, mã sản phẩm và chữ pháp lý phải được đọc lại ở 100%; nếu nguồn
> không còn đủ thông tin thì cần gõ/dựng lại, sau đó duyệt proof trước khi in hàng loạt.

## Dùng nhanh

Mở PowerShell và đứng tại thư mục dự án:

```powershell
cd "C:\Users\Admin\Desktop\RESIZE"
```

Bỏ ảnh vào `INPUT`, rồi chạy một trong các lệnh:

```powershell
# V2 nhanh
.\upscale poster.png 4

# V3 AI chất lượng cao
.\upscale high poster.png 4

# V4 Print có gate native-x4 + PDF/X-4, khổ in rộng 3.000 mm
.\upscale print poster.png 10 --width-mm 3000

# V4 toàn vector cho logo/artwork phẳng
.\upscale vector logo.png 4 --width-mm 3000

# V5 tách layer ở kích thước gốc — nên dùng để chỉnh sửa nhẹ
.\upscale layers poster.png 1

# V5 layer x4 trên master AI V3 — cần bộ model V3
.\upscale layers poster.png 4

# V7 phục dựng toàn ảnh + sửa chữ ở kích thước nguồn — cần V3 CUDA, mở duyệt bằng trình duyệt
.\upscale repair poster.png 1

# V7 sửa chữ rồi làm lớn x4 — cần V3 CUDA
.\upscale repair poster.png 4

# Mở lại màn hình duyệt đơn giản của một bundle V7 (Chrome được ưu tiên)
.\upscale review "C:\duong-dan\poster_V7_REPAIR_x1"
```

Tên file có khoảng trắng phải đặt trong dấu ngoặc kép. Có thể dùng các bí danh `v2`, `v3`, `v4`,
`v5` hoặc `v7`; `v7` tương đương `repair`, còn `vector` là chế độ V4 toàn path riêng.

### Chạy cả thư mục

Truyền đường dẫn thư mục thay cho tên ảnh. Chương trình tìm ảnh trong mọi thư mục con, giữ
cấu trúc thư mục và xử lý tuần tự:

```powershell
.\upscale "D:\BO ANH" 10
.\upscale high "D:\BO ANH" 10
.\upscale print "D:\BO ANH" 4 --width-mm 4000
.\upscale layers "D:\BO ANH" 1
.\upscale repair "D:\BO ANH" 1 --review defer
```

Mỗi ảnh trong batch V4 dùng chung hệ số `n`, khổ rộng, bleed và ICC profile đã truyền.
Mỗi ảnh batch V5 dùng cùng `n`, chạy tuần tự để không tranh VRAM và tạo một bundle riêng.
Batch V7 tự chuyển chế độ duyệt GUI sang `defer`. Mở từng bundle bằng
`.\upscale review "<đường-dẫn-bundle>"` (hoặc bí danh `duyet`), bấm lưu quyết định trên trang cục bộ rồi
chạy lại **đúng lệnh thư mục**. Launcher tự nạp hồ sơ đúng của từng ảnh; người dùng không cần mở hay sửa
JSON. Hai nguồn trùng tên được tách bằng định danh đường dẫn; bundle cũ chỉ bị thay khi manifest chứng
minh đúng nguồn.

Nếu mục tiêu chỉ là làm rõ raster và giữ nguyên chữ/số hiện có, giao diện có nút **Giữ nguyên tất cả
vùng còn lại**. Đây là bulk-keep an toàn: nó không áp dụng chữ OCR đề xuất, không bulk-replace và không
thay giá, số điện thoại hay mã hàng. Sau đó bấm **Lưu và hoàn tất** để đóng phiên; không cần nhập lại chữ.

## Lệnh V5 Smart Layers

```text
.\upscale layers <file-hoặc-thư-mục> <n>
                  [--max-layers 4..60] [--inpaint auto|poster|lama]
                  [--no-semantic] [--allow-huge]
```

- `n=1`: tách/chỉnh ở đúng kích thước nguồn, nhẹ nhất và không cần bộ checkpoint super-resolution V3.
- `1<n<=20`: lấy master AI V3 native x4 đã kiểm định rồi resample một lần tới kích thước yêu cầu;
  vì vậy máy phải có đủ ba checkpoint V3 cục bộ.
- Mặc định tối đa 24 layer tiền cảnh; nền tổng hợp là một layer riêng, nên tổng tối đa là 25.
  Mảnh nhỏ được nhập vào khối cha thay vì tạo hàng trăm “hột”.
- `poster` ưu tiên nền màu/gradient/hình học; `lama` dùng LaMa cho texture ảnh; `auto` chọn bảo thủ
  từ đặc trưng ảnh. Không chế độ nào khôi phục được pixel gốc vốn bị vật thể che.
- `--allow-huge` cho phép vượt cảnh báo 120 MP sau khi tự kiểm tra tài nguyên; hard cap 300 MP vẫn giữ.

Bundle nằm trong `OUTPUT\V5_LAYERS` và luôn gồm OpenRaster `.ora`, ZIP PNG/mask, bản xem, layer map,
contact sheet, OCR JSON và manifest có hash/QA. PSD pixel-layer chỉ được tạo khi mỗi cạnh
không quá 30.000 px và ước tính dữ liệu layer thô không quá 1,6 GB; V5 không tạo PSB giả.
ORA + ZIP là bản mở chuẩn khi PSD vượt giới hạn. ZIP chỉ chứa manifest di động, OCR, các RGBA crop
và mask; nó không đóng gói lặp lại PSD/ORA/preview. OCR chỉ hỗ trợ nhận diện/đặt tên — chữ vẫn là
raster, không giả thành font.

V5 nhận PNG/JPEG/WebP/BMP/TIFF một khung. Ảnh động và TIFF nhiều trang bị từ chối; alpha nguồn hiện
được ghép lên nền trắng trước khi suy luận layer để đầu vào chuẩn hóa không mơ hồ.
Profile sRGB đã chuẩn hóa được nhúng vào mọi PNG màu, PNG màu trong ORA và resource ICC của PSD;
mask alpha `L` cố ý không gắn profile RGB. V5 mở lại để xác minh profile không bị mất khi xuất.

V5 bảo đảm ảnh ghép lại được kiểm tra từ chính các layer 8-bit đã xuất. Tuy vậy, hãy mở PSD/ORA,
tắt từng layer và kiểm tra nền ở 100% trước khi sửa file quan trọng. Canva có thể thay đổi khả năng
nhập PSD; cần thử đúng bundle thật thay vì suy ra từ phần mở rộng.

Mask V5 dùng chính connected-component SAM đã được chọn cho từng vật thể; bán kính inpaint không được
biến thành quyền sở hữu layer. Với chữ raster, V5 trừ pixel giống màu panel ở lòng `O/0`, cửa `G/C` và
khe `N`, phục hồi dấu nhỏ chỉ khi vừa khớp vị trí glyph vừa khớp màu, rồi dùng ViTMatte-S để tạo alpha
phân số trong topology đã khóa. Khi phóng xN, Lanczos chỉ được dùng trong envelope đúng một pixel nguồn,
tránh block nearest-neighbour mà không kéo theo đường panel/vật thể kế bên. `manifest.json` kiểm tra alpha
ngoài vùng cho phép và độ chính xác ghép ảnh độc lập. Alpha x1 đã sạch được khóa làm chuẩn; V5 giải ngược
đúng thứ tự layer để tìm màu nền/tiền cảnh 8-bit khả thi thay vì tăng alpha làm bít lòng chữ. Gate còn đối
chiếu đúng vị trí nét và khoảng âm ở ba ngưỡng alpha; lỗi gate thì bundle không được công bố. Đây vẫn là
raster hữu hạn, không phải SVG/path hay zoom vô cực.

Mở PSD bằng Photoshop/Photopea; mở ORA bằng Krita/GIMP. Với Canva, thử nhập PSD trước; nếu importer
không giữ đúng hierarchy/alpha, hãy upload từng RGBA trong `LAYERS` và đặt theo `canvas_offset` ở
`manifest.json`. Preview chỉ để đối chiếu, không phải master chỉnh sửa. Không có tiêu chuẩn nào khôi
phục được layer graph đã mất từ bitmap phẳng; OpenRaster 0.0.6 chỉ chuẩn hóa cách trao đổi các layer
được V5 suy luận.

Quy trình nhẹ nên dùng: chạy V5 `n=1` → sửa PSD/ORA → xuất một PNG đã ghép từ phần mềm chỉnh sửa →
chạy `upscale high <PNG-đã-sửa> <n>` để lấy ảnh lớn, hoặc `upscale print` để tạo bộ giao in V4.
Nếu cần chính tài liệu layer ở độ phân giải lớn ngay từ đầu, chạy V5 với `n=4`; file sẽ nặng hơn rõ rệt.
PSD/ORA V5 là tài liệu raster RGB trung gian, không phải CMYK/PDF-X giao in và không cam kết giữ DPI
hay khổ vật lý nguồn. Khi giao in, hãy dùng PNG đã sửa với `upscale print ... --width-mm ...` rồi
preflight theo ICC/yêu cầu của nhà in.

## Lệnh V7 Design Repair

```text
.\upscale repair <file-hoặc-thư-mục> <n>
                  [--review gui|defer|auto|strict]
                  [--review-file <TEXT_REVIEW.json>]
                  [--ocr-passes 1..3] [--no-language-model]
                  [--inpaint auto|poster|opencv|strict] [--allow-huge]
```

- `n=1`: phục dựng raster toàn ảnh bằng Swin2SR fidelity native x4 rồi downsample một lần về kích thước
  nguồn; sau đó gỡ/vẽ lại chữ đã duyệt. Chế độ mặc định này cần V3 CUDA, không phải phép copy x1.
- `n=2..20`: dùng cùng dự đoán Swin2SR native x4 và resample một lần tới đúng xN; chữ đã duyệt được vẽ
  trực tiếp ở độ phân giải cuối. V7 không dùng GAN hoặc fusion ba model trong đường phục dựng này.
- Trước Swin2SR, V7 thử deblur luminance có giới hạn, denoise theo mức nhiễu và tăng chi tiết quan sát
  được. Mỗi bước có gate riêng và tự rollback nếu làm xấu ảnh. Nếu SR lỗi hoặc bị gate loại, fallback
  classical/Lanczos phải được ghi trong `_KY_THUAT\QA.json` và `_KY_THUAT\manifest.json`; bundle vẫn ở
  trạng thái cần duyệt, không được đặt tên `01_KET_QUA_DA_DAT.png`.
- `--review gui` là mặc định cho một ảnh. Vùng không chắc chắn mở trang duyệt cục bộ trong Chrome nếu có,
  cho phép nhập chữ đúng, giữ nguyên hoặc để chờ; vùng chưa duyệt không bị tự sửa. `--review defer` phù hợp batch.
- `--review strict` đặt **mọi** vùng, kể cả vùng xanh, về trạng thái chờ cho tới khi review file có quyết
  định rõ ràng. `auto`/`defer` chỉ được tự dựng vùng xanh khi cả PP-OCRv6 và Tesseract tiếng Việt đọc giống
  nhau và confidence riêng của từng engine vượt ngưỡng.
- Cách dễ nhất để duyệt lại là `.\upscale review <bundle>` hoặc `.\upscale duyet <bundle>`. JSON kỹ thuật
  được cất trong `_KY_THUAT`, không cần người dùng mở. Chạy lại ảnh gốc với
  `--review-file <TEXT_REVIEW.json>` chỉ là đường nâng cao; V7 kiểm
  SHA-256 nguồn và fingerprint NFC của bbox + chữ OCR gốc; thiếu hash, sửa nhầm dòng, đổi thứ tự OCR hoặc
  geometry/text không còn khớp đều bị từ chối thay vì gắn quyết định vào vùng khác.
- `--no-language-model` tắt đề xuất chính tả cục bộ nhưng vẫn giữ OCR đối chứng và bước duyệt. Tùy chọn
  này không tắt bước phục dựng raster bằng V3 CUDA.
- Nền dưới nét chữ cũ là phần tổng hợp từ ngữ cảnh nhìn thấy, không phải pixel gốc được khôi phục.
  `strict` có thể giữ nguyên vùng khi ngữ cảnh không đủ tin cậy thay vì buộc nội suy.
- Bbox OCR chỉ dùng để tìm vùng. Khung chữ cuối được siết bằng mask nét thật; font/độ rộng được chọn theo
  hình học và bị `FAILED_QA` nếu tỷ lệ, tâm hoặc kích thước lệch quá ngưỡng. Không font nào được dùng nếu
  thiếu glyph tiếng Việt. Font face, OpenType axes và font size theo tỷ lệ được khóa từ source gate sang
  final gate; SVG lấy đúng render cuối đã khóa đó.

Mỗi ảnh tạo một bundle gọn trong `OUTPUT\V7_REPAIR`:

```text
<ten>_V7_REPAIR_xN\
  01_KET_QUA_DA_DAT.png       chỉ có tên này khi trạng thái PASS
  # hoặc 01_XEM_TRUOC_CAN_DUYET.png khi còn phải duyệt/lỗi QA
  02_SO_SANH.png              bản trước/sau để người dùng tự kiểm tra
  03_CHINH_SUA.svg            nền raster + chữ Unicode <text> chỉnh sửa được
  _KY_THUAT\
    SOURCE.png                ảnh nguồn đã chuẩn hóa
    CLEAN.png                 lớp raster sạch trước khi vẽ lại chữ
    OVERLAY.png               vùng QA
    TEXT_REVIEW.json          quyết định duyệt; giao diện tự đọc/ghi
    QA.json                   gate raster, chữ, seam, clipping và cảnh báo fallback
    manifest.json             hash, model, hạn chế và trạng thái bundle
```

Bundle chỉ có trạng thái `PASS` khi các cổng kiểm tra đạt và không còn vùng bắt buộc duyệt. Trạng thái
`REVIEW_REQUIRED` hoặc `FAILED_QA` là kết quả chẩn đoán/trung gian, không được tự gọi là file giao in.
SVG V7 không nhúng file font, nên máy khác có thể thay font; PNG đánh số `01_...` là bản đối chiếu hình thức.
V7 không tạo PDF/X-4, CMYK, bleed hay khổ mét. Quy trình giao in khuyến nghị là V7 `n=1` → duyệt PNG/SVG
→ dùng `01_KET_QUA_DA_DAT.png` làm đầu vào cho V4 `print` với hệ số và `--width-mm` cần thiết → preflight
theo nhà in.

Super-resolution tạo chi tiết dự đoán có điều kiện từ dữ liệu nhìn thấy; nó không thể chứng minh nội dung
đã mất. Luôn đọc lại chữ, giá, số điện thoại và mã hàng ở 100% trước khi in.

## Lệnh V4 Print

```text
.\upscale print <file-hoặc-thư-mục> <n> [--width-mm <mm>]
                 [--bleed-mm <mm>] [--profile-name <tên-ICC>]
```

- `--width-mm`: chiều rộng thành phẩm dự định, từ 10 đến 100.000 mm. Nếu bỏ qua, V4 suy ra
  kích thước tự nhiên từ DPI của ảnh nguồn.
- `--bleed-mm`: bleed mỗi cạnh ở kích thước **thành phẩm**, từ 0 đến 100 mm; mặc định 0.
  Chương trình tính tỷ lệ trang từ cả khổ thành phẩm lẫn bleed, rồi chia bleed cho đúng mẫu số đó.
- `--profile-name`: tên ICC output profile mà Scribus đã nhận diện.
- `--allow-huge`: cho phép vượt ngưỡng cảnh báo sau khi đã kiểm tra dung lượng; không thể vượt
  hard cap 750 MP bảo vệ Pillow/RAM/ổ đĩa.

`n` nhận từ 2 đến 20. Với V4 `print`, `n` là kích thước cuối của raster đã được QA nhìn thấy và
PNG xem nhanh; sigma USM và Deep candidate đều được quy đổi theo đúng tỷ lệ xN. Các path vector độc lập,
opacity `0` và không phụ thuộc `n`. Vì vậy `n=10` thực sự tạo lớp in 10 lần theo mỗi chiều, không
phải chỉ đổi metadata. Với V4 `vector`, `n` chỉ đổi PNG preview vì path vốn không phụ thuộc pixel.

Mỗi ảnh V4 tạo một bundle trong `OUTPUT\V4_PRINT`:

```text
<ten>_V4_PRINT\
  <ten>_EDITABLE.svg       raster được QA + group path vector ẩn để chỉnh sửa
  <ten>_PRINT_PDFX4.pdf    hình in lấy từ raster được QA, có OutputIntent ICC
  <ten>_PREVIEW_xN.png     PNG render cùng lớp raster để xem/duyệt nội dung
```

Manifest kỹ thuật chứa hash, phiên bản dependency và kết quả QA được cất trong
`APP\manifests\V4_PRINT`, không làm rối thư mục giao file.
Lệnh `vector` dùng cấu trúc tương tự trong `OUTPUT\V4_VECTOR`.

### V4 kiểm tra những gì

V4 chỉ công bố bundle sau khi đã kiểm tra:

- `print`: SVG/PDF tự khai báo đúng guarded-USM, fusion Deep hoặc V3 fallback thực tế, có path vector độc lập
  và opacity mặc định `0`;
- `vector`: SVG/PDF không có image raster;
- cache Deep khớp SHA-256 của ảnh nguồn, master V3 và đúng hệ số xN;
- so trực tiếp từng ứng viên với V3 trên các crop độc lập ở thang pixel native x4: V4 phải giữ
  SSIM/edge, không vượt halo/ringing, tăng ít nhất một chỉ báo edge-strength và một chỉ báo
  energy/detail trên tối thiểu 75% crop; file xN thật còn phải qua gate lần hai;
- Deep phải thắng có ý nghĩa đối chứng no-Deep; nếu không, manifest ghi rõ Deep chỉ là ablation
  diagnostic. Không ứng viên nào đạt thì bundle dùng V3 fallback và cấm tuyên bố V4 tốt hơn;
- PDF 1.6 khai báo PDF/X-4, có OutputIntent/ICC, đủ box, không mã hóa; exporter đo group
  sau khi Scribus nhập và buộc artwork phủ đúng TrimBox + bleed thay vì nằm nhỏ ở góc;
- PNG render lại đạt ngưỡng PSNR, sai số màu và edge F1; lớp vector opacity `0` không được phép
  làm khác hình nhìn thấy so với raster đã được chọn;
- file đầu ra tồn tại, mở được và có đúng kích thước dự kiến.

Đây là **structural self-check**, không phải giấy chứng nhận PDF/X đầy đủ. Trước đơn hàng lớn,
hãy preflight bằng Acrobat Pro/callas/Enfocus hoặc profile GWG Sign & Display của nhà in.

PDF chọn tỷ lệ **1:d** nhỏ nhất để cả `khổ thành phẩm + bleed hai cạnh` không vượt 5.000 mm
trên trang thật; `d = ceil(cạnh ngoài dài nhất / 5.000)`. Ví dụ khổ 9.000 mm không bleed dùng
1:2, còn khổ 100.000 mm với bleed 100 mm dùng 1:21. Kích thước thành phẩm, bleed và tỷ lệ đều
được ghi trong metadata/manifest; phải báo rõ tỷ lệ này cho nhà in.

### Màu in và ICC

Profile mặc định hiện là `ISO Coated v2 300% (basICColor)`. Đây là profile CMYK tham chiếu
phổ biến cho giấy coated, **không phải profile chắc chắn đúng cho bạt/PVC và máy in của bạn**.
Nếu nhà in cung cấp ICC hoặc PDF preset, hãy cài profile vào Scribus rồi truyền đúng tên bằng
`--profile-name`. Luôn yêu cầu proof màu và xác nhận khổ, bleed, tỷ lệ 1:d trước khi in
hàng loạt.

## Phần cứng và runtime

| Máy | V2 Fast | V3 High | V4 Print | V5 Layers | V7 Repair |
|---|---:|---:|---:|---:|---:|
| NVIDIA GTX/RTX phù hợp | Có | Có | Có, bắt buộc cho `print` | Có; nhanh nhất, x1 hoặc xN | Có; bắt buộc mặc định cho x1 và x2..x20 |
| AMD/Intel có Vulkan | Có thể dùng | Không | Chỉ `vector` | Có bằng CPU, chậm; nên x1 | Không ở chế độ phục dựng mặc định |
| Chỉ CPU | Chưa hỗ trợ | Không | Chỉ `vector`, có thể chậm | Có bằng CPU, chậm; nên x1 | Không ở chế độ phục dựng mặc định |

V2 dùng Real-ESRGAN NCNN/Vulkan. V3 dùng PyTorch/CUDA. V4 `print` cần NVIDIA CUDA vì nó tạo/
tái sử dụng master V3 native x4 rồi đánh giá thêm HAT tiled trên **toàn ảnh**. Deep có thể bị loại
khỏi lớp in nếu không thắng guarded-USM no-Deep. Trên RTX 3060
12 GB, ảnh nguồn khoảng `1254×1254`, đầu ra x10 thường cần khoảng **15–25 phút ở lần đầu**;
thời gian thực tế đổi theo tile, model, ổ đĩa, độ phức tạp trace và xuất PDF. Cache được khóa bằng
hash pixel sRGB canonical/master/hệ số xN, nên hai container khác byte nhưng cùng nội dung pixel có thể
tái sử dụng cả V3 x4 lẫn Deep xN; DPI hiện tại vẫn được giữ riêng cho từng job. Cache V3 còn khóa bằng
hash của năm file pipeline, ba model và cấu hình tile; sửa code/model sẽ tự vô
hiệu hóa cache cũ. Đổi nội dung ảnh hoặc đổi `n` thì tầng Deep phải chạy lại. Đây là cache cục bộ một lần,
không phải dịch vụ cloud. V4 `vector` chạy CPU và không cần GPU, nhưng có thể chậm với nhiều path.

Repository công khai chỉ chứa code và tài liệu. Ảnh người dùng, output, `.venv`, model và
binary lớn bị loại khỏi Git. Không sao chép `.venv` sang máy khác. Sau khi clone/pull, cài
toolchain V4 đã pin và kiểm tra SHA bằng:

```powershell
.\APP\engines\V4\setup_v4.ps1
.\APP\engines\V4\setup_v4.ps1 -CheckOnly
```

Lệnh cài tự tải G'MIC 4.0.2, resvg 0.47.0, Scribus 1.6.6 và Python package V4.
Chế độ `print` còn cần runtime/model V3 CUDA; nếu máy không có NVIDIA thì dùng `vector`.

Cài/kiểm tra V5 bằng lệnh riêng. Setup tái dùng môi trường V3 nếu đã có để tránh tốn thêm
vài GB; máy không NVIDIA tự nhận bản PyTorch CPU:

```powershell
.\APP\engines\V5\setup_v5.ps1
.\APP\engines\V5\setup_v5.ps1 -CheckOnly

# Máy bị hạn chế cài Tesseract: bỏ OCR có chủ đích và dùng cùng cờ khi kiểm tra
.\APP\engines\V5\setup_v5.ps1 -SkipTesseract
.\APP\engines\V5\setup_v5.ps1 -CheckOnly -SkipTesseract
```

Setup tải snapshot SAM 2.1/Grounding DINO đã pin, checkpoint LaMa đã kiểm SHA-256 và model
`vie.traineddata` đã pin. Nếu thiếu Tesseract 5, setup dùng WinGet cài đúng gói Windows đã khóa;
máy bị hạn chế cài phần mềm có thể chủ động dùng `-SkipTesseract`, khi đó V5 vẫn chạy bằng SAM/DINO,
bỏ qua vùng OCR và ghi trạng thái vào manifest. Lượt `-CheckOnly` sau đó cũng phải mang cờ
`-SkipTesseract`. Không model hay output lớn nào được commit lên Git.

V7 không bao giờ dùng model English rồi ghi nhãn giả là phiếu tiếng Việt. Nếu thiếu file
`V7\models\tessdata\vie.traineddata` đã kiểm tra, Tesseract được đánh dấu không khả dụng và không thể
đóng góp phiếu xanh độc lập; vùng đó phải qua duyệt.

Máy chỉ có CPU dùng V5 `n=1` (SAM sẽ chậm). Việc tạo master làm nét V3 mới cho `n>1` cần NVIDIA
CUDA; cache V3 đã kiểm định có sẵn vẫn có thể tái dùng, nhưng máy CPU sạch không tự tạo cache đó.

Cài/kiểm tra V7 bằng môi trường OCR CPU tách riêng. `-SkipLanguageModel` bỏ model đề xuất chính tả;
`-SkipTesseract` bỏ phiếu OCR Tesseract và khiến các vùng không được tự xếp xanh chỉ từ một nguồn đọc:

```powershell
.\APP\engines\V7\setup_v7.ps1
.\APP\engines\V7\setup_v7.ps1 -CheckOnly

# Cài tối thiểu khi chỉ muốn OCR Paddle + duyệt tay
.\APP\engines\V7\setup_v7.ps1 -SkipLanguageModel -SkipTesseract
.\APP\engines\V7\setup_v7.ps1 -CheckOnly -SkipLanguageModel -SkipTesseract
```

V7 mặc định luôn gọi checkpoint Swin2SR fidelity đơn qua V3 CUDA, kể cả `n=1`. Bản x1 được suy luận ở
native x4 rồi downsample có kiểm soát; `n=2..20` cũng lấy từ native x4 và resample một lần. Không có GAN/
fusion ba model trong đường V7 này. Nếu model ngôn ngữ cục bộ không có, V7 vẫn chạy OCR/duyệt và ghi rõ
model đề xuất không khả dụng; nội dung không được tự đoán để bù vào.

## Cấu trúc dự án

```text
INPUT/                    ảnh nguồn cục bộ, không đưa lên Git
OUTPUT/V2_FAST/           PNG V2
OUTPUT/V3_HIGH/           PNG V3
OUTPUT/V4_PRINT/          bundle V4 Print SVG/PDF-X-4/PNG
OUTPUT/V4_VECTOR/         bundle toàn vector cho artwork phẳng
OUTPUT/V5_LAYERS/         bundle PSD/ORA/PNG-mask V5
OUTPUT/V7_REPAIR/         bundle V7 phục dựng raster + chữ editable + duyệt/QA
APP/
  upscale_cli.py          bộ điều phối một ảnh và cả thư mục
  engines/V2/             V2 Fast
  engines/V3/             V3 High và kiểm thử
  engines/V4/             V4 Print, export PDF/X-4 và kiểm thử
  engines/V5/             tách layer, tái tạo nền, PSD/ORA và kiểm thử
  engines/V7/             phục dựng raster, OCR/duyệt, sửa chữ và SVG text editable
  manifests/              báo cáo kỹ thuật cục bộ
  masters/                master/cache render cục bộ, có thể tái tạo, không đưa lên Git
  shared/                 công cụ phụ thuộc tải cục bộ, không đưa lên Git
  verification/           kết quả QA máy cục bộ, không đưa lên Git
  work/                   khóa và file công việc tạm; chương trình tự quản lý
upscale.cmd               lệnh duy nhất người dùng cần gọi
VERSION                   phiên bản ứng dụng
CHANGELOG.md              lịch sử thay đổi
```

Chương trình không sửa hoặc xóa ảnh nguồn. Khi chạy lại cùng output, kết quả cũ chỉ được thay
sau khi bundle mới đã render và qua kiểm tra.

## Kiểm thử V4

Chạy test lõi trong môi trường V4, sau đó bắt buộc chạy test Deep trong chính môi trường V3 CUDA.
Biến `RESIZE_V4_REQUIRE_DEEP_RUNTIME=1` biến việc thiếu runtime thành lỗi thay vì báo `skipped`:

```powershell
& .\APP\engines\V4\.venv\Scripts\python.exe -B -m unittest discover `
  -s APP\engines\V4\tests -p "test_v4.py" -v
if ($LASTEXITCODE -ne 0) { throw "V4 core tests failed." }
& .\APP\engines\V4\.venv\Scripts\python.exe -B -m unittest discover `
  -s APP\engines\V4\tests -p "test_cli_safety.py" -v
if ($LASTEXITCODE -ne 0) { throw "V4 CLI safety tests failed." }

$env:RESIZE_V4_REQUIRE_DEEP_RUNTIME = "1"
try {
  & .\APP\engines\V3\.venv\Scripts\python.exe -B -m unittest discover `
    -s APP\engines\V4\tests -p "test_deep_raster_v4.py" -v
  if ($LASTEXITCODE -ne 0) { throw "V4 deep CUDA tests failed or were unavailable." }
}
finally {
  Remove-Item Env:RESIZE_V4_REQUIRE_DEEP_RUNTIME -ErrorAction SilentlyContinue
}
```

## Kiểm thử V5

```powershell
& .\APP\engines\V3\.venv\Scripts\python.exe -B -m unittest discover `
  -s APP\engines\V5\tests -p "test_*.py" -v
.\APP\engines\V5\setup_v5.ps1 -CheckOnly
```

Checklist phát hành còn phải chạy một ảnh thật, mở lại PSD, validate ORA 0.0.6, so ảnh tái ghép
với preview và kiểm tra nền khi tắt/move từng group cha–con.

## Kiểm thử V7

```powershell
& .\APP\engines\V7\.venv\Scripts\python.exe -B -m unittest discover `
  -s APP\engines\V7\tests -p "test_*.py" -v
.\APP\engines\V7\setup_v7.ps1 -CheckOnly
```

Test tự động kiểm tra Unicode NFC tiếng Việt, khóa trường giá/điện thoại/mã hàng, review hash/fingerprint,
phiếu OCR độc lập, footprint nền, pixel ngoài ROI, seam, clipping, font coverage, khóa OpenType fvar
source→final, hình học/baseline SVG, gate/rollback raster, fallback SR, layout bundle thân thiện,
PNG/SVG/manifest và trạng thái QA. Trước khi dùng cho
đơn hàng thật vẫn phải mở PNG/SVG, đọc lại toàn bộ nội dung ở 100% và chạy V4/preflight nhà in riêng.

## Nguồn nền tảng

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) và [Real-ESRGAN NCNN Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
- [Bài báo HAT/CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html), [mã nguồn HAT chính thức](https://github.com/XPixelGroup/HAT), [Swin2SR](https://github.com/mv-lab/swin2sr) và [Spandrel](https://github.com/chaiNNer-org/spandrel)
- [G'MIC](https://gmic.eu/), [VTracer](https://github.com/visioncortex/vtracer), [resvg](https://github.com/linebender/resvg)
- [Scribus](https://www.scribus.net/) và [pikepdf](https://github.com/pikepdf/pikepdf)
- [ISO 15930-7 (PDF/X-4)](https://www.iso.org/standard/55843.html), [PDF Association: yêu cầu PDF/X](https://pdfa.org/technical-side-and-requirements-of-pdfx/), [GWG Sign & Display](https://gwg.org/sign-display/) và [W3C SVG 2 Embedded Content](https://www.w3.org/TR/SVG/embedded.html)
- [SAM 2 chính thức](https://github.com/facebookresearch/sam2), [Grounding DINO chính thức](https://github.com/IDEA-Research/GroundingDINO), [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), [LaMa chính thức](https://github.com/advimman/lama) và [OpenRaster](https://www.openraster.org/baseline/layer-stack-spec.html)
- [PaddleOCR/PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR), [Unicode NFC/UAX #15](https://unicode.org/reports/tr15/), [HarfBuzz](https://harfbuzz.github.io/) và [OpenType](https://learn.microsoft.com/en-us/typography/opentype/spec/)

Chi tiết phiên bản và giấy phép nằm trong `APP\engines`, `APP\engines\V4\SOURCES.md`,
`APP\engines\V5\SOURCES.md`, `APP\engines\V7\SOURCES.md` và
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Giấy phép

Phần code riêng của dự án được phát hành theo [MIT License](LICENSE), bản quyền 2026
`nguyenduytamgithub`. Thành phần, binary và model bên thứ ba tuân theo giấy phép riêng của
từng dự án.
