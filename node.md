Dữ Liệu 
  ┌────────────┬────────────────────────────────┬────────────────────────────────────────┐
  │ Thuộc tính │ Ý nghĩa                        │ Giá trị                                │
  ├────────────┼────────────────────────────────┼────────────────────────────────────────┤
  │ age        │ Tuổi học sinh                  │ 15 đến 22                              │
  │ sex        │ Giới tính                      │ F = nữ, M = nam                        │
  │ address    │ Khu vực sống                   │ U = thành thị, R = nông thôn           │
  │ Medu       │ Trình độ học vấn của mẹ        │ 0–4                                    │
  │ Fedu       │ Trình độ học vấn của bố        │ 0–4                                    │
  │ traveltime │ Thời gian đi từ nhà đến trường │ 1–4                                    │
  │ studytime  │ Thời gian học mỗi tuần         │ 1–4                                    │
  │ failures   │ Số lần trượt môn trước đó      │ 0–4                                    │
  │ schoolsup  │ Hỗ trợ học tập từ trường       │ yes, no                                │
  │ famsup     │ Hỗ trợ học tập từ gia đình     │ yes, no                                │
  │ paid       │ Có học thêm trả phí            │ yes, no                                │
  │ activities │ Tham gia hoạt động ngoại khóa  │ yes, no                                │
  │ higher     │ Muốn học lên cao               │ yes, no                                │
  │ internet   │ Có internet ở nhà              │ yes, no                                │
  │ romantic   │ Có quan hệ tình cảm            │ yes, no                                │
  │ famrel     │ Chất lượng quan hệ gia đình    │ 1–5                                    │
  │ freetime   │ Thời gian rảnh sau giờ học     │ 1–5                                    │
  │ goout      │ Mức độ đi chơi với bạn bè      │ 1–5                                    │
  │ health     │ Tình trạng sức khỏe            │ 1–5                                    │
  │ absences   │ Số buổi vắng học               │ Số nguyên, trong file hiện khoảng 0–32 │
  │ G1         │ Điểm kỳ 1                      │ 0–10, do đã quy đổi                    │
  │ G2         │ Điểm kỳ 2                      │ 0–10, do đã quy đổi                    │
  │ G3         │ Điểm cuối kỳ cần dự đoán       │ 0–10, do đã quy đổi                    │
  └────────────┴────────────────────────────────┴────────────────────────────────────────┘

  Giải thích thêm các thang:

  Medu/Fedu:
  0 = không có
  1 = tiểu học
  2 = THCS
  3 = THPT
  4 = đại học/cao hơn

  traveltime:
  1 = dưới 15 phút
  2 = 15–30 phút
  3 = 30 phút–1 giờ
  4 = trên 1 giờ

  studytime:
  1 = dưới 2 giờ/tuần
  2 = 2–5 giờ/tuần
  3 = 5–10 giờ/tuần
  4 = trên 10 giờ/tuần

  famrel/freetime/goout/health:
  1 = rất thấp/rất kém
  5 = rất cao/rất tốt

Thuật toán 
  ┌─────────────────────────┬──────────────────────────────────────┐
  │ Mô hình                 │ Thuộc nhóm hồi quy nào               │
  ├─────────────────────────┼──────────────────────────────────────┤
  │ Linear Regression       │ Hồi quy tuyến tính đa biến           │
  │ Ridge Regression        │ Hồi quy tuyến tính có regularization │
  │ Decision Tree Regressor │ Mô hình cây quyết định hồi quy       │
  │ Random Forest Regressor │ Mô hình Rừng ngẫu nhiên dạng hồi quy │
  │ KNN Regressor           │ Hồi quy dựa trên láng giềng gần nhất │
  └─────────────────────────┴──────────────────────────────────────┘  

1. Linear Regression 
  > Trong đề tài này, em sử dụng mô hình Linear Regression, cụ thể là hồi quy tuyến tính đa biến. Đây là một
  > thuật toán học có giám sát, dùng để dự đoán một giá trị số liên tục dựa trên nhiều thuộc tính đầu vào.
  >
  > Với bài toán của em, giá trị cần dự đoán là điểm cuối kỳ G3. Các thuộc tính đầu vào gồm những thông tin
  > như tuổi, thời gian học, số lần vắng học, điểm kỳ 1 G1, điểm kỳ 2 G2 và một số yếu tố liên quan đến học
  > tập, gia đình, sinh hoạt của học sinh.
  >
  > Nguyên lý hoạt động của mô hình là tìm ra một công thức tuyến tính biểu diễn mối quan hệ giữa các thuộc
  > tính đầu vào và điểm G3. Công thức tổng quát có dạng:
  >
  > G3 = a1*x1 + a2*x2 + ... + an*xn + b
  >
  > Trong đó x1, x2, ..., xn là các thuộc tính đầu vào, còn a1, a2, ..., an là các hệ số mà mô hình học được
  > trong quá trình huấn luyện. Mỗi hệ số thể hiện mức độ ảnh hưởng của một thuộc tính đến điểm dự đoán.
  >
  > Khi huấn luyện, mô hình sẽ dự đoán điểm G3 trên dữ liệu train, sau đó so sánh với điểm G3 thực tế để tính
  > sai số. Mục tiêu của mô hình là tìm bộ hệ số sao cho sai số giữa điểm thực tế và điểm dự đoán là nhỏ
  > nhất. Trong thư viện scikit-learn, mô hình LinearRegression sử dụng phương pháp bình phương tối thiểu,
  > tức là tối thiểu hóa tổng bình phương sai số.
  Về cơ chế vận hành, cấu phần LinearRegression của scikit-learn sử dụng Phương pháp bình phương tối thiểu (OLS). Thuật toán này áp dụng đại số ma trận để tính toán một mạch ra bộ trọng số tối ưu nhằm tối thiểu hóa tổng bình phương sai số,
  W  = (XT x X)-1 x Y



  Ridge Regression

  > Thuật toán thứ hai em sử dụng là Ridge Regression. Đây cũng là một mô hình hồi quy tuyến tính, nhưng có
  > thêm kỹ thuật regularization để hạn chế hiện tượng overfitting.
  >
  > Về cơ bản, Ridge Regression vẫn dự đoán điểm G3 dựa trên công thức tuyến tính:
  >
  > G3 = a0 + a1*x1 + a2*x2 + ... + an*xn
  >
  > Trong đó x1, x2, ..., xn là các thuộc tính đầu vào như thời gian học, số buổi vắng, điểm G1, điểm G2, còn
  > a0, a1, ..., an là các hệ số mà mô hình cần học.
  >
  > Điểm khác biệt của Ridge so với Linear Regression là khi huấn luyện, mô hình không chỉ cố gắng giảm sai
  > số dự đoán, mà còn thêm một phần phạt vào các hệ số. Phần phạt này làm cho các hệ số không bị quá lớn.
  >
  > Có thể hiểu đơn giản là nếu Linear Regression chỉ tập trung tìm bộ hệ số sao cho sai số nhỏ nhất, thì
  > Ridge Regression vừa muốn sai số nhỏ, vừa muốn các trọng số bị quá lớn , quá nhạy cảm khiến overtiing 
  >
  > Điều này giúp mô hình ổn định hơn, đặc biệt khi các thuộc tính đầu vào có liên quan mạnh với nhau, ví dụ
  > G1 và G2. Nếu hệ số quá lớn, mô hình có thể học quá sát dữ liệu train và dự đoán kém trên dữ liệu mới.
  >
  > Ưu điểm của Ridge Regression là giảm overfitting, ổn định hơn Linear Regression trong một số trường hợp,
  > và vẫn giữ được tính dễ giải thích của mô hình tuyến tính. Nhược điểm là mô hình vẫn giả định quan hệ
  > tuyến tính, nên nếu dữ liệu có quan hệ phi tuyến mạnh thì Ridge cũng chưa xử lý tốt.
  LOSS = MSE + alpha + ( w1**2 + w1**2)

  Decision Tree Regressor

  > Thuật toán thứ ba em sử dụng là Decision Tree Regressor, tức cây quyết định cho bài toán hồi quy. Khác
  > với Linear Regression, mô hình này không cố gắng tìm một công thức đường thẳng, mà chia dữ liệu thành
  > nhiều nhánh dựa trên các điều kiện của thuộc tính.
  >
  > Với bài toán dự đoán điểm G3, cây quyết định có thể học các quy tắc dạng như: nếu điểm G2 cao thì điểm G3
  > có xu hướng cao, nếu số lần trượt môn nhiều thì điểm dự đoán có thể thấp hơn, hoặc nếu số buổi vắng cao
  > thì kết quả học tập có thể bị ảnh hưởng.
  >
  > Trong quá trình huấn luyện, mô hình sẽ chọn thuộc tính và ngưỡng chia sao cho sau khi chia, các mẫu trong
  > cùng một nhóm có điểm G3 càng gần nhau càng tốt. Với bài toán hồi quy, mỗi nút lá của cây sẽ đưa ra một
  > giá trị dự đoán, thường là trung bình điểm G3 của các mẫu thuộc nút đó.
  >
  > Ví dụ, nếu một học sinh mới đi theo các nhánh điều kiện của cây và rơi vào một nút lá, mô hình sẽ lấy giá
  > trị dự đoán ở nút lá đó làm điểm G3 dự đoán.
  >
  > Ưu điểm của Decision Tree là dễ hiểu, dễ trực quan hóa và có thể mô hình hóa các quan hệ phi tuyến giữa
  > dữ liệu đầu vào và điểm G3. Tuy nhiên, nhược điểm là cây quyết định rất dễ bị overfitting nếu cây quá
  > sâu, tức là mô hình học quá chi tiết dữ liệu train và dự đoán kém trên dữ liệu test.

  Random Forest Regressor

  > Thuật toán thứ tư em sử dụng là Random Forest Regressor. Đây là mô hình ensemble, tức là kết hợp nhiều mô
  > hình nhỏ lại với nhau. Cụ thể, Random Forest gồm nhiều cây quyết định hồi quy.
  >
  > Nếu Decision Tree chỉ dùng một cây để dự đoán, thì Random Forest tạo ra nhiều cây khác nhau. Mỗi cây được
  > huấn luyện trên một phần dữ liệu hoặc một tập thuộc tính khác nhau, nhờ đó các cây có thể học được các
  > góc nhìn khác nhau từ dữ liệu.
  >
  > Khi cần dự đoán điểm G3 cho một học sinh mới, mỗi cây trong rừng sẽ đưa ra một giá trị dự đoán riêng. Sau
  > đó Random Forest lấy trung bình các dự đoán của tất cả các cây để cho ra kết quả cuối cùng.
  >
  > Nguyên lý này giúp mô hình ổn định hơn so với một cây quyết định đơn lẻ. Vì nếu một cây dự đoán hơi lệch,
  > các cây khác có thể bù lại, làm cho kết quả tổng thể đáng tin cậy hơn.
  >
  > Với bài toán của em, Random Forest phù hợp vì dữ liệu có nhiều thuộc tính khác nhau, và mối quan hệ giữa
  > các yếu tố như điểm G1, G2, số buổi vắng, thời gian học với điểm G3 có thể không hoàn toàn tuyến tính.
  >
  > Ưu điểm của Random Forest là thường cho kết quả dự đoán tốt, giảm overfitting so với Decision Tree đơn
  > lẻ, xử lý được quan hệ phi tuyến và có thể đánh giá mức độ quan trọng của các thuộc tính. Nhược điểm là
  > mô hình khó giải thích hơn Linear Regression và Decision Tree, đồng thời thời gian huấn luyện cũng lâu
  > hơn do phải xây dựng nhiều cây.

  Kết quả:

  ┌─────────────────────────┬────────┬────────┬────────┬────────┐
  │ Mô hình                 │    MAE │    MSE │   RMSE │     R2 │
  ├─────────────────────────┼────────┼────────┼────────┼────────┤
  │ Ridge Regression        │ 0.3700 │ 0.3437 │ 0.5862 │ 0.8590 │
  │ Linear Regression       │ 0.3704 │ 0.3440 │ 0.5865 │ 0.8589 │
  │ Random Forest Regressor │ 0.3794 │ 0.3946 │ 0.6282 │ 0.8381 │
  │ Decision Tree Regressor │ 0.4692 │ 0.9731 │ 0.9864 │ 0.6009 │
  └─────────────────────────┴────────┴────────┴────────┴────────┘

  Nếu cô hỏi mô hình nào tốt nhất, bạn trả lời:

  > Dựa trên kết quả đánh giá, mô hình tốt nhất là Ridge Regression. Vì mô hình này có MAE, MSE, RMSE thấp
  > nhất và R2 Score cao nhất trong các mô hình được so sánh. Cụ thể, Ridge Regression có RMSE = 0.5862 và R2
  > = 0.8590, tốt hơn một chút so với Linear Regression.

  Nói ngắn gọn:

  Tốt nhất: Ridge Regression
  Dựa vào: MAE, MSE, RMSE thấp nhất và R2 cao nhất

  MAE

  > MAE đo trung bình mô hình dự đoán sai lệch bao nhiêu điểm so với điểm thật. Ví dụ MAE = 0.37 nghĩa là
  > trung bình mỗi dự đoán lệch khoảng 0.37 điểm so với điểm thực tế.

  MSE

  > MSE cũng đo sai số giữa điểm thật và điểm dự đoán, nhưng nó bình phương sai số lên. Vì vậy những lỗi dự
  > đoán lớn sẽ bị phạt nặng hơn. MSE càng thấp thì mô hình càng ít mắc lỗi lớn.

  RMSE

  > RMSE là căn bậc hai của MSE. Nó cũng đo mức sai lệch dự đoán, nhưng quay về cùng đơn vị với điểm số. Ví
  > dụ RMSE = 0.586 nghĩa là sai số điển hình của mô hình khoảng 0.586 điểm.

  R2 Score

  > R2 đo mức độ mô hình giải thích được sự biến động của điểm G3. R2 càng gần 1 thì mô hình càng phù hợp dữ
  > liệu. Ví dụ R2 = 0.859 nghĩa là mô hình giải thích được khoảng 85.9% sự thay đổi của điểm G3.

  Tóm gọn khi nói với cô:

  > MAE và RMSE cho biết mô hình dự đoán lệch trung bình khoảng bao nhiêu điểm. MSE nhấn mạnh mạnh hơn các
  > lỗi lớn. Còn R2 cho biết mô hình giải thích được bao nhiêu phần trăm sự biến động của điểm cần dự đoán.