# Flow và quy tắc của agent

Tài liệu này mô tả **kiến trúc và hành vi mục tiêu** của agent. Luật game chi tiết nằm ở
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

## Nguyên tắc lập kế hoạch theo ngày

Agent dùng **một kế hoạch đầy đủ cho cả ngày**, thay vì tạo route rồi chèn các
hành động cứu hộ hoặc tối ưu cục bộ trong lúc đang thực thi.

Pipeline trách nhiệm vẫn tách theo module:

1. `planner.py` chọn target kinh tế cho các ô cần xét.
2. `farm_tasks.py` chuyển trạng thái vật lý hiện tại + target thành **toàn bộ task
   cần làm trong ngày**.
3. `scheduler.py` chọn số hand, phân task và dựng route/queue cho toàn bộ ngày.
4. Runtime thực hiện queue đã chốt. Runtime không tự nghĩ thêm WATER/FEED/CARE/
   HARVEST chỉ vì đang đi qua một ô hoặc vì task chưa được làm ở đầu ngày.

Mọi task bắt buộc của ngày chỉ cần **hoàn thành trước refresh cuối ngày**. Không
có khái niệm crop phải được HARVEST trước một hour/step cụ thể trong cùng ngày;
không dùng `max_lifespan_step` để tạo intraday deadline hoặc ép chen route.
Scheduler được tự do sắp thứ tự task trong ngày để giảm travel và số hand, miễn
đúng dependency và hoàn thành toàn bộ mandatory work trước rollover.

Các dependency phải được giữ:

- `PLANT → WATER` cho cây mới.
- `PLACE → FEED/CARE` khi lịch của con vật mới yêu cầu.
- `PICKUP → action` đối với vật tư cần mang theo.
- `HARVEST → quay về shed → DROP → SELL` khi cần biến hàng thành cash trong
  ngày cuối.

## Flow mỗi ngày

1. Giới hạn ngày kết thúc theo `end_day` và lượt cuối có thể hành động trong
   `configuration.episodeSteps`. Observation terminal không có lượt thực thi.
2. Phát hiện đất vừa mở, đưa vào danh sách chờ chọn target; sau lần mua đất
   thành công đầu tiên thì chuyển vĩnh viễn từ opening sang planner.
3. **Hour 0:** cập nhật target, tạo toàn bộ daily task từ observation hiện tại,
   ước lượng hand/vật tư, bán hàng dư và gửi lệnh mua/thuê. Worker trả `PASS`.
4. Ở observation đầu tiên sau khi purchases/hires đầu ngày đã được xác nhận,
   xây lại daily task set từ state thật rồi dựng **final daily queues**.
   Nếu một top-up HIRE vẫn cần thiết thì chỉ chốt queue sau khi số hand thực tế
   dùng cho plan đã được xác nhận.
5. **Phần còn lại của ngày:** thực hiện queue đã chốt. Không prepend rescue
   route, không append từng task mới vào worker đang chạy và không opportunistic
   HARVEST chỉ vì worker tình cờ đứng trên tile có sản phẩm.
6. Mỗi lượt vẫn kiểm tra legality của action và số vật tư thực tế trước khi gửi
   action cho engine. Market có thể SELL hàng đã về shed và thực hiện các order
   đã nằm trong daily plan.
7. Chỉ replan giữa ngày khi observation chứng minh daily plan không còn khả thi.
   Khi replan, dựng lại **toàn bộ phần task chưa hoàn thành** từ vị trí hiện tại
   của mọi worker; không vá plan bằng một route cứu hộ cục bộ rồi quay lại route cũ.

Worker actions chạy **trước market orders**. Hạt giống, con vật, thức ăn và
hand mua ở lượt này chỉ dùng được từ observation tiếp theo. Daily plan không
được giả định một market order đã thành công trước khi observation xác nhận.

## Khi nào được replan giữa ngày

Replan là recovery cho divergence thực tế, không phải cơ chế scheduling bình
thường. Các trường hợp hợp lệ gồm:

- HIRE không thành công hoặc hand spawn khác vị trí dự đoán.
- BUY bị thiếu tiền, bị market-order cap cắt hoặc vật tư thực nhận khác plan.
- `PICKUP` không lấy đủ lượng đã dự kiến.
- Tile/state thay đổi khiến action đã lên lịch không còn legal hoặc không còn
  cần thiết.
- Một producer/structure biến mất hoặc thay đổi ngoài giả định của daily plan.

Khi replan:

- Xác định lại task nào đã hoàn thành và task nào còn lại.
- Dùng vị trí + inventory thật của worker tại observation hiện tại.
- Pack lại toàn bộ mandatory work trước, rồi optional work.
- Không yêu cầu worker quay về vị trí trước khi replan.
- Không giữ prefix/suffix cũ nếu route mới tốt hơn và vẫn bảo toàn dependency.

Đất mới mở hoặc target mới xuất hiện giữa ngày mặc định được xử lý ở daily plan
kế tiếp. Chỉ đưa vào ngày hiện tại nếu agent chủ động chạy một global replan và
chứng minh toàn bộ mandatory work hiện có vẫn hoàn thành được.

## Opening và mua đất

- 25 ô NW ban đầu: **12 MELON, 9 WHEAT, 2 COW, 2 SHEEP**.
  Opening không cam kết bón phân; phân được bán để tài trợ sản xuất.
- Từ ngày index 2, chọn hai ô WHEAT có sản phẩm, ưu tiên gần shed,
  đổi target lần lượt sang COW và SHEEP. Việc đặt thực tế còn phụ thuộc
  thu hoạch, vật tư, tiền và capacity của daily plan.
- Chính sách gửi một `BUY_LAND` ở hour 0 của mỗi **ngày index 7 và 10**,
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
- Producer vật lý đang tồn tại phải được chăm sóc theo state thật, độc lập với
  việc target kinh tế tương lai của tile là gì. Target quyết định replacement/
  investment, không được làm một cây/con vật đang sống biến mất khỏi daily task set.
- Xét ô gần shed trước, cập nhật cung và lao động sau mỗi lựa chọn.
  Chuồng/pasture trống chỉ nhận loại con vật phù hợp.
- Đầu ngày thêm các ô vào danh sách xét lại; xử lý theo batch để giữ decision
  deterministic và latency ổn định khi mở 75–100 ô đất.

## Tạo daily task

`schedules.py` là nguồn lịch WATER/FERTILIZE/FEED/CARE/HARVEST theo **tuổi**
(`day - planted_day` hoặc `day - placed_day`), không theo ngày lịch.

`farm_tasks.py` phải quét cả target và producer vật lý hiện tại để tạo đầy đủ
công việc của ngày trước khi scheduler route:

- Cây mới luôn có `PLANT → WATER` trong cùng daily plan.
- Cây có `consecutive_unwatered >= 1` và chưa WATER hôm nay phải có WATER
  mandatory trong daily task set, kể cả hôm nay không nằm trong cadence chuẩn.
- Con vật có `consecutive_unfed >= 1` và chưa FEED hôm nay phải có FEED
  mandatory trong daily task set, kể cả cadence chuẩn không yêu cầu FEED.
- Animal output chỉ tạo HARVEST khi cadence/chính sách hiện tại yêu cầu hoặc
  output có nguy cơ clip. HARVEST này được đưa vào daily plan ngay từ đầu;
  runtime không chen HARVEST khi worker tình cờ đi qua pen.
- FEED/CARE cuối mùa dùng cùng horizon với forecast. Không FEED ngày cuối nếu
  không còn refresh hữu ích; bỏ CARE khi bonus chỉ tới sau endpoint.
- Physical producer không bị bỏ qua chỉ vì `target is None` hoặc target đã đổi.
- Task không còn hợp lệ sau observation mới được loại khi global replan chạy;
  không giữ stale action chỉ để bảo toàn queue cũ.

Survival không có priority “làm ngay”. WATER/FEED cứu producer là mandatory vì
phải hoàn thành **trong ngày**, nhưng scheduler vẫn có thể đặt chúng ở cuối một
route nếu route đó chắc chắn hoàn thành trước rollover.

## Scheduler và nhân công

Scheduler nhận toàn bộ daily task set và giải bài toán assignment/route theo
capacity của cả ngày.

- Farmer và hand bắt đầu làm sau pass/mua đầu ngày có số lượt thực thi còn lại
  theo observation thật; không giả định 24 action nếu đã mất hour 0.
- Mandatory work phải được pack trước optional work.
- Nếu số worker hiện tại không thể hoàn thành mọi mandatory task trong ngày,
  `hands_needed()` tăng số hand trong giới hạn vốn và `MAX_HANDS`.
- Optional work chỉ được giữ khi còn capacity hoặc giá trị tăng thêm lớn hơn
  chi phí thuê Fibonacci biên.
- Giá trị optional task lấy từ cash dự kiến; investment mới dùng remaining
  forecast output trừ vốn/vật tư tương lai. Planner đã xét labor khi chọn target,
  không chia lại lợi nhuận cho số ngày chăm sóc.
- Khi so các route cùng tập task/số hand, ưu tiên tổng travel/pickup thấp hơn và
  khả năng đưa hàng cần bán về shed tốt hơn.
- Gom các FEED/CARE/COLLECT_FERTILIZER gần nhau khi làm giảm route thực tế;
  không ép worker quay về vị trí xuất phát giữa các task.
- Việc đánh giá optional task nên dựa trên **incremental route cost** của task
  trong route hiện có, không chỉ khoảng cách standalone từ spawn đến tile.
- Không dùng `max_lifespan_step` như intraday scheduling deadline cho crop.
- Không dành thêm turn cho một HARVEST “có thể phát sinh” ngoài daily plan.
- Không dùng local rescue insertion để bảo vệ survival; survival đã nằm trong
  mandatory daily task set.

Nếu scheduler không thể pack toàn bộ mandatory work ngay cả tại giới hạn hand,
đó là planning failure cần được thấy rõ trong test/metrics; không che lỗi bằng
runtime rescue route.

## Bảo vệ action khi thực thi

Runtime guard chỉ bảo vệ correctness của action đã được plan:

- `PICKUP` không được lấy nhiều hơn lượng thật trong shed.
- `PLANT/PLACE` không gửi nếu item thật chưa tồn tại hoặc horizon không còn cho
  phép bắt đầu.
- Chặn `DIG/PLANT` trên COOP/PASTURE hoặc ô có con vật.
- HARVEST stale/duplicate không được gửi nếu tile hiện không có output.
- Nếu guard làm một action không thể thực thi và dependency sau nó không còn
  đúng, đánh dấu daily plan invalid và global replan phần còn lại.

Runtime **không**:

- prepend WATER/FEED/HARVEST rescue route;
- quay worker về vị trí cũ sau rescue;
- opportunistic HARVEST chỉ vì worker đang đứng trên producer;
- append từng investment/maintenance task mới vào queue đang chạy;
- thay đổi priority theo hour chỉ vì mandatory task chưa được làm sớm.

## Bán hàng và quản lý vốn

- Bán sản phẩm dư trong shed; giữ vật tư đã được daily plan dành cho PICKUP.
- WHEAT phải đủ cho FEED nằm trong daily plan và buffer hợp lý cho ngày kế tiếp,
  nhưng không giữ/mua dự trữ sau endpoint của mùa.
- Không mua WHEAT cho FEED ngày cuối nếu không còn refresh sau đó.
- Đầu ngày còn giữ WHEAT cho con vật mà daily plan thực sự có thể mua + PLACE.
- Dự tính tiền bán theo giá từng đơn vị; chỉ `SELL` tạo doanh thu.
  Tiền đối thủ và cạnh tranh market thực tế vẫn có thể làm kết quả khác dự báo.
- Ngoài opening: mua feed cần thiết, bảo vệ ngân sách mandatory hires, rồi mua
  animal/seed cho investment đã được daily plan nhận.
- Không BUY animal chỉ vì có target; phải có placement trong daily plan hoặc
  một committed future placement policy được planner định giá rõ ràng.
- `DROP` giữa ngày chỉ nằm trong route khi giúp cashflow/capacity và không làm
  mất khả năng hoàn thành mandatory work.

## Ngày cuối và liquidation

Ngày cuối là ngoại lệ duy nhất có ràng buộc execution boundary khác rollover
thông thường: hàng chỉ có giá trị reward nếu kịp biến thành cash trước terminal.

- Tạo HARVEST cho producer có output nếu expected cash sau travel/return là dương.
- Không làm maintenance chỉ tạo lợi ích sau terminal.
- Task tạo hàng để bán phải đủ thời gian thực hiện, quay về shed, `DROP` và còn
  một decision để `SELL` trước observation terminal.
- Optional terminal work được chọn theo incremental cash sau chi phí labor và
  route; ưu tiên value thực tế, không chỉ khoảng cách.
- Fertilizer cuối ngày cũng chỉ có giá trị nếu kịp về shed và SELL.
- Không mua feed/seed/animal ở cuối mùa nếu không còn production refresh có thể
  tạo reward trước terminal.

`liquidation.py` có thể cắt/rebuild terminal routes để bảo đảm cashout, nhưng
không dùng logic này làm mô hình scheduling cho các ngày bình thường.

## Phân công module

| Module | Trách nhiệm |
| --- | --- |
| `expansion_agent.py` | Điều phối daily planning, thực thi queue và kích hoạt global replan khi plan invalid |
| `opening_book.py` | Portfolio mở đầu, chuyển đổi WHEAT, lịch mua đất |
| `planner.py`, `horizon.py` | Chọn target kinh tế và giới hạn đầu tư theo horizon |
| `forecast.py`, `labor.py` | Dự báo sản lượng, dòng tiền và chi phí lao động |
| `farm_tasks.py` | Tạo toàn bộ daily task, nhu cầu vật tư và lệnh mua |
| `scheduler.py` | Chọn hand, assignment và route cho toàn bộ daily task set |
| `intraday.py` | Công cụ hỗ trợ replan/reconciliation; không chèn rescue task vào queue bình thường |
| `liquidation.py` | Terminal cashout |
| `schedules.py`, `products.py` | Lịch chăm sóc và danh mục/giá vốn chung |
| `selling.py` | Bán dư và giữ vật tư đã commit |

## Quy tắc khi sửa repo

- Giữ tính độc lập của state giữa các trận; không dùng biến global mutable
  để giữ kế hoạch của agent.
- Giá vốn/loại sản phẩm lấy từ `products.py`; lịch chăm sóc ở `schedules.py`;
  giới hạn cửa sổ ở `horizon.py`. Không sao chép bảng sang module khác.
- Thay đổi lịch chăm sóc phải kiểm tra bằng `experiments.crop_schedules`
  và `experiments.animal_yields` trên interpreter.
- Chạy `.venv/bin/python -m pytest -q` sau thay đổi logic; giữ regression test
  cho opening, daily mandatory completion, survival, tồn kho, terminal cashout
  và global-replan behavior.
- Test phải phát hiện mandatory task còn sót ở end-of-day thay vì dựa vào một
  runtime rescue để che scheduling failure.
- Đo latency bằng `.venv/bin/python -m experiments.benchmark_agent --opponent self`;
  báo p95, lượt chậm nhất và số lượt vượt `actTimeout` theo diện tích đất.
  Benchmark mặc định dùng ngưỡng 0,75 giây; có thể đổi bằng `--timeout`.
- So sánh giữ target/ngày khởi tạo bằng `experiments.scheduler_comparison`.
  Lưu source baseline trước khi sửa, dùng seed trong `result.json` vì engine
  có thể ghi `configuration.seed = null` trong replay. Kết quả có thiếu/thừa
  ngày PLANT/PLACE/BUILD/DIG không được coi là cải thiện routing thuần.
- Khi đánh giá scheduler, theo dõi riêng movement actions, useful actions,
  hire cost, mandatory misses, crop death, animal escape và stranded inventory.
- Cập nhật tài liệu này khi đổi chiến lược hoặc thứ tự xử lý.
- Replay/output sinh ra để trong `replays/` (được Git bỏ qua).
- Agent hiện giả định board 10×10, 24 lượt/ngày, market cap 10 và lịch chăm sóc
  cho mùa chuẩn 30 ngày. Truyền configuration khác không đồng nghĩa hỗ trợ
  đầy đủ mọi cấu hình; công thức endpoint chỉ chặn vượt lượt cuối.