# KIẾN TRÚC V4 TỐI GIẢN — Phân chia trách nhiệm duy nhất

Tài liệu này là nguồn chuẩn duy nhất cho kiến trúc V4, thay thế mọi diễn giải kiến trúc cũ.

## 3. Phân chia trách nhiệm duy nhất

### Bộ tìm vùng chữ

Chỉ chịu trách nhiệm:
- đọc video một lần;
- phát hiện vùng chữ;
- ghép các vùng qua thời gian thành từng vệt;
- giữ tọa độ và thời gian của mỗi vệt.

Bộ này không được quyết định nội dung và không được tự gọi một vùng là phụ đề.

### Gemini

Chỉ chịu trách nhiệm:
- nhìn ảnh toàn cảnh và ảnh cắt của vệt;
- chọn mã vệt nào là phụ đề;
- trả kết quả có/không theo mã vệt.

Gemini không được trả tọa độ, không tạo vùng che và không sửa hình học.

### Bộ dựng video

Chỉ chịu trách nhiệm:
- nhận danh sách mã vệt đã được xác nhận;
- lấy tọa độ đã có từ bộ tìm vùng chữ;
- tạo vùng che;
- đặt bản dịch và xuất video.

Bộ dựng video không được tự phân loại lại.

## 4. Luồng xử lý sản phẩm — chỉ còn 7 bước

1. Đọc video một lần và giữ chung khung hình cho các bước sau.
2. Tìm vùng chữ và ghép thành vệt; mỗi vệt có mã, tọa độ và thời gian.
3. Lọc rẻ tại máy bằng một bộ lọc duy nhất: ngoài vùng đáy → loại; quá ngắn hoặc không ổn định → loại; còn lại → đưa sang Gemini.
4. Gom các vệt còn lại thành 2–3 bảng ảnh và gửi Gemini. Gemini chỉ trả danh sách mã vệt là phụ đề.
5. Lưu một quyết định cho mỗi vệt, không lưu một quyết định cho mỗi cửa sổ mask.
6. Tạo mask từ tọa độ gốc của các vệt có quyết định dương; sau đó mới đọc chữ và dịch.
7. Xuất video và kiểm tra tối thiểu: video tồn tại, số khung đúng, mask đều trỏ tới vệt có quyết định dương, không có đường tạo mask khác.

Dạng rút gọn:

Video → tìm chữ và tọa độ → lọc rẻ → Gemini chọn mã vệt → che theo tọa độ cũ → đọc/dịch → xuất video

## 5. Sổ quyết định tối giản

Chỉ cần một bản ghi cho mỗi vệt:

{"run_id": "round35-sample5", "track_id": "T003", "is_subtitle": true, "model": "gemini-3.5-flash-lite", "response_ref": "responses/request_01.json", "cache_hit": false}

Quy tắc:
- track_id duy nhất trong một lần chạy.
- Mask chỉ cần mang track_id; không tạo mã ngẫu nhiên riêng cho từng mask.
- Một quyết định dương của T003 cấp quyền cho mọi cửa sổ hình học thuộc T003.
- Thiếu bản ghi, khác lần chạy, quyết định âm hoặc file phản hồi không tồn tại -> không render mask.
- Kết quả dùng lại phải ghi rõ; không được trình bày như kết quả vừa gọi mới.

## 6. Những phần phải xóa khỏi đường chạy sản phẩm

- Đường tạo mask từ kết quả đọc chữ.
- Bất kỳ chỗ nào tự gán is_subtitle=true hoặc tên model mặc định.
- Một quyết định riêng cho từng mask; vòng 34 có 259 mask nhưng chỉ cần quyết định theo 13-14 vệt.
- Nhiều loại mã nguồn gốc cho cùng một quyết định.
- Nhiều sổ ghi chồng lên nhau.
- Bộ đọc chữ có quyền tạo vùng che.
- Chương trình báo cáo tự dựng lại quyết định.
- Dữ liệu kiểm định trộn vào dữ liệu chạy video.
- Cơ chế chia nhóm/thử lại phức tạp trong đường chính; chỉ được thử lại tối đa một lần khi phản hồi hỏng định dạng hoặc hết thời gian.
- Các mô-đun rỗng hoặc không có tác dụng: r2_language, behavior_discriminator và mọi lớp tương tự nếu chưa thực sự tham gia quyết định.
- Đường dự phòng âm thầm. Lỗi GPU, lỗi Gemini hoặc lỗi đọc phản hồi phải báo rõ.

## 7. Những phần chỉ thuộc bộ kiểm định, không thuộc sản phẩm

Các mục sau được chuyển ra thư mục kiểm định và không chạy khi xử lý video thông thường:

- bộ 47 track;
- 8 mẫu âm chỉ định;
- canary N001;
- chạy thử nhóm 10 rồi chia nhóm 5;
- confusion matrix;
- pháp y phản hồi;
- so sánh baseline;
- đo coverage;
- ảnh khung xấu nhất;
- báo cáo vòng và đóng gói bằng chứng.

Bộ kiểm định chỉ được đọc đầu ra thật của pipeline. Nó không được tạo quyết định, sửa nhãn hay thêm mask.
