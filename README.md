# Local Print Image Upscaler — V2, V3 và V4 Print

Chương trình xử lý ảnh cục bộ trên Windows cho bảng hiệu, poster và file in khổ lớn.
Ảnh nguồn không được gửi lên dịch vụ đám mây.

## Bốn chế độ

| Chế độ | Đầu ra chính | Khi nên dùng |
|---|---|---|
| **V2 Fast** | PNG phóng lớn | Cần kết quả nhanh, máy có Vulkan |
| **V3 High** | PNG AI chất lượng cao | Ảnh chụp hoặc ảnh nhiều texture, máy có NVIDIA CUDA |
| **V4 Print** | Raster AI/USM đã qua ablation + SVG/PDF/X-4 + PNG | Bản in nghiêm túc, cần phục hồi đồng đều toàn ảnh bằng NVIDIA CUDA |
| **V4 Vector** | SVG/PDF/X-4 toàn path | Logo, chữ, mảng màu phẳng; không dùng cho ảnh chụp/gradient |

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
```

Tên file có khoảng trắng phải đặt trong dấu ngoặc kép. Có thể dùng `v2`, `v3` hoặc `v4`;
`vector` là chế độ V4 toàn path riêng, không phải bí danh của `print`.

### Chạy cả thư mục

Truyền đường dẫn thư mục thay cho tên ảnh. Chương trình tìm ảnh trong mọi thư mục con, giữ
cấu trúc thư mục và xử lý tuần tự:

```powershell
.\upscale "D:\BO ANH" 10
.\upscale high "D:\BO ANH" 10
.\upscale print "D:\BO ANH" 4 --width-mm 4000
```

Mỗi ảnh trong batch V4 dùng chung hệ số `n`, khổ rộng, bleed và ICC profile đã truyền.

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

| Máy | V2 Fast | V3 High | V4 Print |
|---|---:|---:|---:|
| NVIDIA GTX/RTX phù hợp | Có | Có | Có, bắt buộc cho `print` |
| AMD/Intel có Vulkan | Có thể dùng | Không | Chỉ `vector` |
| Chỉ CPU | Chưa hỗ trợ | Không | Chỉ `vector`, có thể chậm |

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

## Cấu trúc dự án

```text
INPUT/                    ảnh nguồn cục bộ, không đưa lên Git
OUTPUT/V2_FAST/           PNG V2
OUTPUT/V3_HIGH/           PNG V3
OUTPUT/V4_PRINT/          bundle V4 Print SVG/PDF-X-4/PNG
OUTPUT/V4_VECTOR/         bundle toàn vector cho artwork phẳng
APP/
  upscale_cli.py          bộ điều phối một ảnh và cả thư mục
  engines/V2/             V2 Fast
  engines/V3/             V3 High và kiểm thử
  engines/V4/             V4 Print, export PDF/X-4 và kiểm thử
  manifests/              báo cáo kỹ thuật cục bộ
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

## Nguồn nền tảng

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) và [Real-ESRGAN NCNN Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
- [Bài báo HAT/CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html), [mã nguồn HAT chính thức](https://github.com/XPixelGroup/HAT), [Swin2SR](https://github.com/mv-lab/swin2sr) và [Spandrel](https://github.com/chaiNNer-org/spandrel)
- [G'MIC](https://gmic.eu/), [VTracer](https://github.com/visioncortex/vtracer), [resvg](https://github.com/linebender/resvg)
- [Scribus](https://www.scribus.net/) và [pikepdf](https://github.com/pikepdf/pikepdf)
- [ISO 15930-7 (PDF/X-4)](https://www.iso.org/standard/55843.html), [PDF Association: yêu cầu PDF/X](https://pdfa.org/technical-side-and-requirements-of-pdfx/), [GWG Sign & Display](https://gwg.org/sign-display/) và [W3C SVG 2 Embedded Content](https://www.w3.org/TR/SVG/embedded.html)

Chi tiết phiên bản và giấy phép nằm trong `APP\engines`, `APP\engines\V4\SOURCES.md` và
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Giấy phép

Phần code riêng của dự án được phát hành theo [MIT License](LICENSE), bản quyền 2026
`nguyenduytamgithub`. Thành phần, binary và model bên thứ ba tuân theo giấy phép riêng của
từng dự án.
