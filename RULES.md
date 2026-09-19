# Flow và quy tắc của agent

Tài liệu này mô tả **hành vi đang được triển khai**. Luật game chi tiết nằm ở
[docs/README.md](docs/README.md); observation/action API ở
[docs/AGENTS.md](docs/AGENTS.md).

## Điểm vào và dữ liệu

`agents.expansion_agent.make_agent(end_day=29, seed=0)` tạo một agent có state
riêng cho mỗi trận. Agent nhận `(obs, configuration=None)` và trả về
`{"farmer": [...], "hands": [[...]], "market": [[...]]}`.

- `targets[(x, y)] = (name, fertilize) | None`: loại cây/con vật dự kiến cho ô.
- `plans`: hàng đợi hành động của farmer, rồi các hand theo thứ tự observation.
- State còn giữ ngày, số hand cần thuê, vật tư dành trước, vị trí đang chờ
  trồng/đặt, hand chưa xác nhận vị trí spawn và đất mới chưa được lên lịch.
- Quyết định là deterministic theo observation; `seed` được giữ để tương thích
  với caller, không dùng để random trong agent.

## Flow mỗi lượt

1. Giới hạn ngày kết thúc theo `end_day` và lượt cuối có thể hành động trong
   `configuration.episodeSteps`. Observation terminal không có lượt thực thi.
2. Phát hiện đất vừa mở, đưa vào danh sách chờ chọn target; sau lần mua đất
   thành công đầu tiên thì chuyển vĩnh viễn từ opening sang planner.
   Định giá tối đa 2 ô/lượt, tiếp tục phần còn lại ở lượt sau; không đổi target
   của ô đang có công việc trong queue. Ô mới chưa định giá tạm thời để trống.
3. **Hour 0:** cập nhật target, tạo task để tính nhu cầu nhân công/vật tư,
   reset kế hoạch ngày, bán hàng dư rồi gửi lệnh mua/thuê. Worker trả `PASS`.
4. **Hour 1:** dựa vào tiền, vật tư và hand thực sự đã có để xây queue;
   có thể thuê bù với ngân sách 22 lượt cho hand xuất hiện ở hour 2.
5. **Các lượt trong ngày:** kiểm tra spawn của hand mới; thêm việc ở ô trống
   khi đủ thời gian và vốn; giao việc chăm sóc còn thiếu cho worker rảnh.
   Tính lại vật tư dành trước từ các lệnh `PICKUP` còn trong queue.
6. Lấy một action/worker, áp dụng kiểm tra hạt giống, vật tư, hạn trồng,
   ưu tiên thu hoạch con vật sẵn sàng và bảo vệ chuồng/pasture.
7. Bán hàng, mua vật tư cho công việc khả thi, rồi thuê thêm phục vụ mở rộng.
   Chỉ gửi 10 market orders đầu tiên.

Worker actions chạy **trước market orders**. Hạt giống, con vật, thức ăn và
hand mua ở lượt này chỉ dùng được từ observation tiếp theo. Do đó agent
lập queue sau lượt mua đầu ngày, không giả định lệnh mua đã thành công.

## Opening và mua đất

- 25 ô NW ban đầu: **12 MELON, 9 WHEAT, 2 COW, 2 SHEEP**.
  Opening không cam kết bón phân; phân được bán để tài trợ sản xuất.
- Từ ngày index 2, chọn hai ô WHEAT có sản phẩm, ưu tiên gần shed,
  đổi target lần lượt sang COW và SHEEP. Việc đặt thực tế còn phụ thuộc
  thu hoạch, vật tư, tiền và lịch worker.
- Chính sách hiện tại gửi một `BUY_LAND` ở hour 0 của mỗi **ngày index 7 và 10**,
  trước các khoản chi tùy chọn. Đây là lịch cố định trong `LAND_BUY_DAYS`,
  không phải quyết định theo tỷ lệ lấp đầy đất.
- Nếu không đủ tiền, engine bỏ qua lệnh; không tự thử lại vào ngày khác.
  Opening chỉ kết thúc khi observation xác nhận đã mở thêm quadrant.

## Chọn target và tính lợi nhuận

- Cùng một cửa sổ cho mọi ứng viên: từ `day` đến
  `min(day + 16, end_day)`, tính cả thu hoạch ngày cuối.
- Không bắt đầu cây/con vật có lần cho sản phẩm đầu tiên ngoài cửa sổ.
- Điểm = dòng tiền thị trường tăng thêm − vốn mua − chi phí nhân công tăng thêm.
  Dòng tiền đã tính thức ăn, phân bón và tác động cung lên giá từng đơn vị.
  **Không chia lợi nhuận cho số ngày**, không áp quota vốn cây/con vật.
- Chỉ chọn điểm dương. Khi bằng điểm: ưu tiên loại đang có ít hơn,
  sau đó dùng thứ tự tuple target để kết quả ổn định.
- Forecast gồm sản lượng còn lại theo tuổi/trạng thái thực, hàng trong shed
  và inventory, sản lượng đối thủ quan sát được và nhu cầu shop hiện mở.
  Không biết quyết định tương lai hoặc inventory riêng của đối thủ.
- Cây được tái trồng chỉ khi lịch thu hoạch tiếp theo còn nằm trong cửa sổ;
  tính tiền từng lần mua seed. Cam kết bón phân đi cùng target suốt chu kỳ.
- Không đổi target của con vật đã đặt hoặc cây chưa hết chu kỳ. Giữ chuyển
  đổi đã được opening lên lịch nếu còn kịp cho sản phẩm.
- Xét ô gần shed trước, cập nhật cung và lao động sau mỗi lựa chọn.
  Chuồng/pasture trống chỉ nhận loại con vật phù hợp.
- Đầu ngày thêm các ô vào danh sách xét lại; mỗi lượt xử lý tối đa 2 ô đủ
  điều kiện, dùng observation mới nhất. Các target đang chờ vẫn góp vào
  forecast. Giới hạn theo số ô giữ quyết định deterministic và giảm thời gian
  mỗi action khi mở 75–100 ô đất; đây không phải bảo đảm thời gian trên mọi máy.

## Lập lịch và bảo vệ hành động

- `schedules.py` là nguồn lịch WATER/FERTILIZE/FEED/CARE theo **tuổi**
  (`day - planted_day` hoặc `day - placed_day`), không theo ngày lịch.
- FEED/CARE của vật nuôi tính thêm tuổi vào ngày kết thúc thực tế. Giữ FEED
  cần cho lần sinh sản phẩm cuối có thể thu hoạch và để tránh bỏ đi; không
  FEED ngày cuối. Trong ba ngày cuối, bỏ CARE nếu bonus chỉ tới sau mùa.
  Trước cửa sổ này giữ nhịp CARE để không xáo trộn chỗ dành cho tái trồng.
  Kiểm tra việc còn thiếu và seller dùng cùng cửa sổ chăm sóc;
  buyer bỏ nhu cầu hiện tại sau hạn FEED, vẫn giữ buffer ngày kế tiếp bảo thủ.
  Forecast dùng cùng horizon FEED/CARE và cùng cadence HARVEST với runtime;
  ép thu sản phẩm còn giữ vào ngày cuối cửa sổ dự báo.
  `experiments.animal_horizon` đối chiếu 90 trường hợp 1–30 ngày với engine:
  cùng sản phẩm/phân bón và không có con vật bỏ đi trong thời gian chơi.
- Cây mới phải có `PLANT → WATER` trong ngày. Hai lần refresh liên tiếp
  thiếu nước/thức ăn làm cây thành cỏ hoặc con vật bỏ đi.
- Cây có `consecutive_unwatered >= 1` được tạo WATER dù ngoài lịch tuổi;
  task cứu cây đứng trước mọi nhóm khác, kể cả harvest con vật. Trong opening,
  giữ thứ tự các lượt WATER đúng lịch đã được dự trù để bảo toàn chuỗi
  tài trợ; WATER cứu cây ngoài lịch vẫn được đẩy lên trước.
- Scheduler ưu tiên task bắt buộc, task khẩn cấp, sản phẩm con vật sẵn sàng, chăm sóc và
  deadline cây; gộp pickup, tính đường đi và chỉ nhận task vừa ngân sách.
- Khi thử chèn task, chỉ đếm số bước di chuyển/action/pickup; dựng queue
  thực thi sau khi chọn xong. Dự báo lao động tái sử dụng khoảng cách đến shed.
- Sau opening, task chăm sóc được thử thêm tối đa ba thứ tự xếp (việc xa/dài,
  theo cột, theo hàng). Chỉ nhận phương án xếp đủ toàn bộ task; cùng số hand
  thì ưu tiên lượng hàng có thể về kho trong ngày. Không thay route đầu tư
  chứa PLANT/PLACE/DIG/BUILD, vì buyer intraday đang dựa vào các cam kết này.
- Khi dựng queue hour 1 và kho đủ WHEAT cho toàn bộ task, gom FEED trong
  tối đa hai lượt duyệt. Thử chuyển từng task FEED/CARE/thu phân bón sang
  worker đã mang WHEAT, chèn sau các task khác của worker nhận. Điểm chọn
  bằng tổng bước bắt buộc cộng 4 lượt cho mỗi worker mang WHEAT; chỉ nhận
  nếu điểm giảm, hàng bị giữ ngoài kho không tăng, tuyến nhận đủ lượt về
  kho và dư một lượt. Giữ nguyên phân công/thứ tự tương đối của việc khác,
  số hand và tập task chưa xếp được. Không gom trong opening hoặc thiếu kho.
- Giữ nguyên phân công khi chỉ mở NW. Bỏ qua số hand chắc chắn không đủ ngay
  cả khi không phải di chuyển. Phương án thay thế có sản phẩm con vật dành
  thêm một lượt cho HARVEST phát sinh khi worker đi qua ô có sản phẩm.
  Đây là tìm kiếm heuristic có số lần thử cố định, không chứng minh tối ưu toàn cục.
- Farmer và hand có từ hour 1 có 23 lượt. Worker đang làm giữa ngày chỉ có
  `24 - hour` lượt, trừ queue đã cam kết và thời gian chờ hàng nếu cần.
- Trần 16 hand/ngày. Sau opening, thử từng số hand với route thực tế:
  bảo vệ WATER/FEED chống chết, animal output đã chạm held-cap và harvest
  có deadline; WATER/FEED/CARE định kỳ còn lại được định giá như task tùy chọn;
  chỉ mua thêm capacity tùy chọn khi giá trị công việc tăng thêm lớn hơn
  giá thuê Fibonacci biên. Không thuê nếu task bắt buộc còn lại không thể
  được cải thiện và phần tùy chọn không bù giá thuê. Hand đã thuê là sunk cost.
  Opening giữ sizing cũ để bảo toàn chuỗi mua vật tư/thu phân, ngoại trừ ngày cuối.
- Giá trị task tùy chọn lấy giá bán từng đơn vị theo stock hiện tại. Đầu tư
  mới dùng sản lượng forecast trừ vốn/vật tư tương lai; planner đã xét chi
  phí lao động khi chọn target, không chia thêm cho số ngày chăm sóc. WATER
  định kỳ được tính tối thiểu một đơn vị crop; FEED/CARE dùng sản phẩm và
  phân dự kiến trừ giá WHEAT. Chưa mô
  phỏng toàn bộ tác động giá giữa các task cùng ngày. Thuê mở rộng giữa ngày
  cũng chọn theo giá trị task đã được cấp vật tư trừ chi phí thuê.
  Dự báo lao động là ước lượng greedy, không phải route tương lai chính xác.
- Spawn dự đoán sai: xóa queue của hand đó và lên lại từ vị trí quan sát được.
  Lệnh thuê thất bại/bị cắt phải giải phóng task và vật tư đã dành trước.
- Không giao trùng ô đã có action trong queue. Thêm việc trong ngày phải giữ
  việc đã cam kết; áp dụng phần task xếp được, phần còn lại thử ở lượt sau.
- Thiếu seed: giữ `PLANT` đầu queue, worker `PASS`. `PICKUP` chỉ lấy lượng
  còn thực sự trong shed; thiếu bao nhiêu thì giữ lại lệnh lấy phần đó.
- Quá hạn bắt đầu: bỏ `PLANT/PLACE` cùng các action chăm sóc kế ngay sau,
  giữ đường đi và các task khác.
- Đứng trên con vật có sản phẩm: ưu tiên `HARVEST`, đưa action bị hoãn về
  queue. Chỉ một worker/ô áp dụng ưu tiên này trong cùng lượt.
- Chặn `DIG/PLANT` trên COOP/PASTURE hoặc ô có con vật, kể cả cấu trúc vừa
  được worker trước đó trong danh sách xây/đặt trong cùng lượt.

## Bán hàng và quản lý vốn

- Bán sản phẩm dư trong shed mỗi lượt; giữ vật tư cho pickup đang cam kết.
  Riêng WHEAT phải giữ đủ feed còn thiếu hôm nay và lịch feed ngày mai,
  nhưng không giữ/mua dự trữ cho ngày sau endpoint của mùa.
- Đầu ngày còn giữ WHEAT cho con vật mà tiền sau bán hàng có thể mua được.
- Dự tính tiền bán theo giá từng đơn vị; chỉ `SELL` tạo doanh thu.
  Tiền đối thủ và cạnh tranh market thực tế vẫn có thể làm kết quả khác dự báo.
- Ngoài opening: mua feed, bảo vệ ngân sách thuê cần thiết, rồi mua animal/seed.
  Trong opening: ưu tiên vật tư cho portfolio trước thuê tùy chọn.
- Mua cho ô trống phải được intraday scheduler chấp nhận về thời gian/vốn.
  Vật tư đi trước các lệnh thuê mở rộng để tránh thuê người không có việc.
- `DROP` giữa ngày giúp bán và tái đầu tư khi còn thời gian; không hy sinh
  task cần thiết chỉ để về kho. Kho đầy làm mất hàng; inventory worker không
  tránh được giới hạn shed khi tự chuyển hàng cuối ngày.
- Ngày cuối: giữ prefix queue dài nhất còn đủ thời gian về kho và DROP, chừa
  một decision sau DROP để bán. Observation terminal không thực thi SELL;
  hàng còn trong inventory ở đó không tăng cash reward. `liquidation.py`
  thực hiện bước này trước khi lấy action của worker.
- Ngày cuối tạo HARVEST cho mọi cây có sản lượng trong tập target, kể cả
  chưa hết chu kỳ; chỉ WATER nếu tăng ngay sản lượng cây một lần, không
  FERTILIZE hay tái trồng. Thu hoạch mới là
  task tùy chọn, nhưng khi nhận phải đủ thời gian travel/HARVEST/return/DROP
  và chừa SELL; giá thuê thêm phải được tiền thu dự kiến bù đắp.
- Mọi task tạo hàng ngày cuối, kể cả FERTILIZER, phải đủ ngân sách quay về
  để giá trị dùng quyết định thuê phản ánh hàng bán được. Task thu phân là
  tùy chọn, không ép thuê nếu giá trị không bù giá thuê biên. Chính sách định
  giá mới có thể đổi target/ngày đầu tư; báo riêng tính hợp lệ lịch khởi tạo
  trong phép so sánh khóa replay, không coi lịch lệch là cải thiện thuần routing.

## Phân công module

| Module | Trách nhiệm |
| --- | --- |
| `expansion_agent.py` | Điều phối lượt và state |
| `opening_book.py` | Portfolio mở đầu, chuyển đổi WHEAT, lịch mua đất |
| `planner.py`, `horizon.py` | Chọn target và giới hạn thời gian |
| `forecast.py`, `labor.py` | Dự báo sản lượng, dòng tiền và chi phí lao động |
| `farm_tasks.py` | Task, nhu cầu vật tư và lệnh mua |
| `scheduler.py`, `intraday.py`, `liquidation.py` | Route/queue, thuê người, thêm việc và bán hàng trước terminal |
| `schedules.py`, `products.py` | Lịch chăm sóc và danh mục/giá vốn chung |
| `selling.py` | Bán dư và giữ feed |

## Quy tắc khi sửa repo

- Giữ tính độc lập của state giữa các trận; không dùng biến global mutable
  để giữ kế hoạch của agent.
- Giá vốn/loại sản phẩm lấy từ `products.py`; lịch chăm sóc ở `schedules.py`;
  giới hạn cửa sổ ở `horizon.py`. Không sao chép bảng sang module khác.
- Thay đổi lịch chăm sóc phải kiểm tra bằng `experiments.crop_schedules`
  và `experiments.animal_yields` trên interpreter.
- Chạy `.venv/bin/python -m pytest -q` sau thay đổi logic; giữ regression về
  opening, deadline, survival, tồn kho và intraday scheduling.
- Đo latency bằng `.venv/bin/python -m experiments.benchmark_agent --opponent self`;
  báo p95, lượt chậm nhất và số lượt vượt `actTimeout` theo diện tích đất.
  Benchmark mặc định dùng ngưỡng 0,75 giây; có thể đổi bằng `--timeout`.
- So sánh giữ target/ngày khởi tạo bằng `experiments.scheduler_comparison`.
  Lưu source baseline trước khi sửa, dùng seed trong `result.json` vì engine
  có thể ghi `configuration.seed = null` trong replay. Kết quả có thiếu/thừa
  ngày PLANT/PLACE/BUILD/DIG không được coi là cải thiện hợp lệ.
- Cập nhật tài liệu này khi đổi chiến lược hoặc thứ tự xử lý.
- Replay/output sinh ra để trong `replays/` (được Git bỏ qua).
- Agent hiện giả định board 10×10, 24 lượt/ngày, market cap 10 và lịch chăm sóc
  cho mùa chuẩn 30 ngày. Truyền configuration khác không đồng nghĩa hỗ trợ
  đầy đủ mọi cấu hình; công thức endpoint chỉ chặn vượt lượt cuối.
