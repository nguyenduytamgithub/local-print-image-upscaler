# Local Print Image Upscaler — V2 Fast & V3 High

Trình phóng lớn ảnh chạy cục bộ trên Windows, hướng tới ảnh bảng hiệu, poster và
file in khổ lớn. Ảnh nguồn không được gửi lên dịch vụ đám mây.

Phiên bản hiện tại có hai chế độ:

- **V2 Fast**: Real-ESRGAN NCNN/Vulkan + G'MIC, ưu tiên tốc độ và tài nguyên vừa phải.
- **V3 High**: phối hợp Swin2SR, HAT và Real-ESRGAN trên CUDA, ưu tiên chất lượng.

> Upscale làm ảnh dễ sử dụng hơn ở kích thước lớn nhưng không thể khôi phục chắc
> chắn mọi chi tiết vốn không tồn tại trong ảnh nguồn. Với chữ nhỏ, giá tiền và
> thông tin pháp lý, luôn xem ở 100% và duyệt bản in thử trước khi sản xuất.

## Dùng nhanh

Mở PowerShell tại thư mục dự án:

```powershell
cd "C:\duong-dan\local-print-image-upscaler"
```

Bỏ ảnh vào `INPUT`, sau đó chạy:

```powershell
# V2 nhanh, một ảnh
.\upscale poster.png 4

# V3 chất lượng cao, một ảnh
.\upscale high poster.png 4

# V2 cho cả thư mục và mọi thư mục con
.\upscale "D:\BO ANH" 10

# V3 cho cả thư mục và mọi thư mục con
.\upscale high "D:\BO ANH" 10
```

`n` là hệ số của cả chiều rộng và chiều cao, nhận từ `2` đến `20`. Kết quả nằm
trong `OUTPUT/V2_FAST` hoặc `OUTPUT/V3_HIGH`. Khi xử lý thư mục, chương trình:

- dùng cùng một hệ số cho toàn bộ ảnh;
- nhận PNG, JPEG, WebP, BMP và TIFF một khung hình;
- giữ lại cấu trúc thư mục con;
- xử lý tuần tự để tránh tranh VRAM;
- bỏ qua riêng ảnh hỏng và tiếp tục các ảnh còn lại;
- không sửa hoặc xóa ảnh nguồn.

## Phần cứng

| Máy | V2 Fast | V3 High |
|---|---:|---:|
| NVIDIA GTX/RTX, driver phù hợp | Có | Có |
| AMD hoặc Intel có Vulkan | Có thể dùng | Không |
| Chỉ CPU, không có GPU/Vulkan | Chưa hỗ trợ | Không |

V2 dùng bản Real-ESRGAN NCNN/Vulkan nên không cần PyTorch/CUDA, nhưng vẫn cần
GPU hoặc iGPU có driver Vulkan hoạt động. V3 hiện yêu cầu GPU NVIDIA CUDA; mức
VRAM thực tế còn phụ thuộc kích thước ảnh và tile.

## Cài đặt trên máy khác

Repository công khai chỉ nên chứa code và tài liệu. Nó **không** chứa ảnh người
dùng, kết quả render, môi trường `.venv`, CUDA, model hoặc binary lớn. Vì Python
virtual environment gắn với đường dẫn máy đã tạo ra nó, không nên sao chép
nguyên `.venv` sang máy khác.

Bản `0.1.0` hiện chưa có bootstrap installer công khai hoàn chỉnh, vì vậy chỉ
`git clone` hoặc `git pull` chưa đủ để chạy trên một máy sạch. Hướng đóng gói dự
kiến là một lệnh `setup` có các bước:

1. nhận diện NVIDIA CUDA hoặc Vulkan;
2. cài riêng thành phần V2/V3 phù hợp;
3. tải dependency và model từ nguồn chính thức;
4. kiểm tra SHA-256;
5. chạy kiểm tra `doctor` và một ảnh mẫu nhỏ.

Không nên tự động cài driver GPU vì việc đó có thể cần quyền quản trị và khởi
động lại Windows. Installer chỉ nên phát hiện và hướng dẫn rõ driver còn thiếu.

## Cấu trúc dự án

```text
INPUT/                 ảnh nguồn cục bộ, không đưa lên Git
OUTPUT/                ảnh kết quả cục bộ, không đưa lên Git
APP/
  upscale_cli.py       bộ điều phối một ảnh và cả thư mục
  engines/V2/          mã nguồn V2 Fast
  engines/V3/          mã nguồn V3 High và kiểm thử
upscale.cmd            lệnh duy nhất người dùng cần gọi
VERSION                phiên bản ứng dụng
CHANGELOG.md           lịch sử nâng cấp
```

Master x4, manifest kỹ thuật, báo cáo QA, runtime và model đều là dữ liệu cục bộ
và bị loại khỏi Git bằng `.gitignore`.

## Quản lý phiên bản và file nặng

- Source code dùng Semantic Versioning, ví dụ `0.1.0`, `0.2.0`, `1.0.0`.
- Mỗi bản phát hành dùng Git tag cùng số phiên bản và cập nhật `CHANGELOG.md`.
- Mỗi model được khóa bằng tên file và SHA-256 riêng.
- Binary/runtime nên được tải từ nguồn chính thức hoặc đóng gói trong GitHub
  Releases; không commit trực tiếp vào lịch sử Git.
- Không dùng Git làm kho lưu ảnh input/output hay bản sao `.venv`.

GitHub cảnh báo file lớn hơn 50 MiB và chặn file lớn hơn 100 MiB trong Git thông
thường, vì vậy repository nguồn cần được giữ gọn.

## Nguồn nền tảng

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Real-ESRGAN NCNN Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
- [Swin2SR](https://github.com/mv-lab/swin2sr)
- [HAT](https://github.com/XPixelGroup/HAT)
- [Spandrel](https://github.com/chaiNNer-org/spandrel)
- [G'MIC](https://gmic.eu/)
- [PyTorch](https://pytorch.org/)

Chi tiết nguồn model, hash và giấy phép bên thứ ba nằm trong `APP/engines` và
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Giấy phép

Phần code riêng của dự án được phát hành theo [MIT License](LICENSE), bản quyền
2026 `nguyenduytamgithub`. Các thành phần và model bên thứ ba vẫn tuân theo giấy
phép riêng được ghi trong `THIRD_PARTY_NOTICES.md` và các thư mục license tương
ứng.
