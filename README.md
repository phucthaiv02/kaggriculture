# Kaggriculture Expanding Heuristic

Agent deterministic dùng để kiểm tra chính xác toàn bộ lifecycle trồng trọt và
chăn nuôi trước khi thêm strategic policy.

## Phạm vi hiện tại

- Bắt đầu sản xuất trong quadrant `NW`; sau khi thanh toán nhu cầu vận hành,
  bán sản phẩm và mở quadrant 5×5 kế tiếp (`NE`, giá `$1,000`) ngay khi số dư
  trong cùng market turn đủ tiền. Toàn bộ 25 ô NE sau đó trở thành một cohort
  crop mới, được chọn theo profit/day và khả năng mua đủ seed + farmhand.
- Layout khởi đầu là `12 MELON`, `9 WHEAT`, `2 COW`, `2 SHEEP`; crop thu hoạch
  xong được thay bằng loại có projected profit/day cao nhất.
- Số farm hand được suy ra từ số route tối thiểu có thể hoàn thành trong 24
  turn. Animal có route riêng để bảo đảm `FEED + COLLECT_FERTILIZER` mỗi ngày;
  các crop route được phân hoạch độc lập.
- Tự mua seed/con giống/thức ăn, xây chuồng, trồng, tưới, cho ăn, chăm sóc,
  thu hoạch, thu fertilizer và bán sản phẩm.
- Không có model, checkpoint hoặc dependency ML.

Layout cố định (`W/S` là ô Wheat đồng thời tiếp cận shed):

```text
       x=0  x=1  x=2  x=3  x=4
y=0     M    M    M    M    M
y=1     M    M    M    M    M
y=2     M    M    W    W   Sheep
y=3     W    W    W   Cow  Sheep
y=4     W    W    W   Cow   W/S
```

Wheat và vật nuôi ban đầu nằm gần shed vì cần phục vụ thường xuyên; Melon có chu
kỳ dài nằm ở hai hàng xa. Sau harvest, loại crop trên từng vị trí có thể đổi.
Các route NE dùng cửa shed `(5,4)` và được phân hoạch riêng với NW để tránh hành
trình xuyên quadrant và giữ thời gian lập kế hoạch ổn định.
Chi phí thuê và buffer thức ăn hai ngày được giữ trước khi mua thêm seed/con
giống để tài sản đang sống không chết vì cạn vốn vận hành.

Luật game đầy đủ nằm trong [docs/README.md](docs/README.md), hướng dẫn chạy và
submit nằm trong [docs/AGENTS.md](docs/AGENTS.md).

## Chạy kiểm tra

```bash
python -m pytest
```

Chạy một season local:

```bash
PYTHONPATH=src python scripts/smoke_game.py
```

Đóng gói submission thuần Python:

```bash
python scripts/package_submission.py
```

Archive được tạo tại `submissions/heuristic_agent.tar.gz`.

## Dynamic crop selection

Ở `hour=0` của ngày một cohort one-time crop đến hạn thu hoạch, policy dự phóng
từng loại crop trên window `[day, 29]`, giả sử toàn bộ ô vừa thu hoạch chỉ lặp
lại loại đó và giá bán giữ ở snapshot hiện tại:

```text
profit/day = (projected sale revenue - all seed costs) / remaining days
```

- One-time crop lặp chu kỳ theo `Time to Max Yield` và yield tối đa khi tưới,
  không tính fertilizer: Wheat 4, Carrot 3, Melon 6.
- Ongoing crop dùng lịch yield Tomato `8,9,10,11` và Strawberry `10,12,14,16`,
  mỗi lần yield 1 vì crop planner hiện chưa fertilize.
- Chỉ tính cycle/yield có ngày harvest `<= 29`. Crop không kịp harvest nhận 0.
- Nếu không crop nào có projected profit dương, cohort được thu hoạch rồi để
  trống; không mua seed vô ích ở cuối season.

Các loại được xếp hạng theo profit/day, nhưng planner chỉ chọn loại mà toàn bộ
lệnh mua seed cho cohort còn đủ tiền sau khi giữ trước chi phí farmhand dự kiến
và thức ăn cho hai ngày. `BUY_SEED` đủ số lượng được phát ngay trong market
orders của hour 0. Route sau đó chạy chuỗi
`WATER → HARVEST → PLANT replacement → WATER`. Ràng buộc mua đủ cả cohort ngăn
trường hợp chỉ một phần tile được trồng lại vì hết tiền giữa chừng.

## Dynamic route planner

`generate_unit_clusters()` không chứa danh sách cluster viết tay. Với tối đa 12
target, planner xét các tập route khả thi và chọn partition có ít route nhất;
tie-break bằng tổng số turn. Mỗi route phải thỏa:

```text
shortest travel từ shed + PICKUP dùng chung + service actions <= 24 - hour
```

Đường đi trong một route là shortest open path: unit xuất phát gần shed nhưng
không phải quay lại vì inventory được tự chuyển vào shed cuối ngày. Bốn animal
gần shed dùng một route riêng và một manifest Wheat/con giống dùng chung.
Layout crop lớn hơn 12 target dùng phép gộp gần nhất rồi tái phân phối asset để
loại các route thừa mà vẫn giữ ngân sách.

Mỗi turn, observation được chuyển lại thành các task hiện hành. Unit chọn task
trong route dựa trên deadline priority, quãng đường và inventory đang mang. Vì
vậy sau `WATER`, `FEED` hoặc `HARVEST`, công việc kế tiếp được lập lại từ state
thật thay vì tiếp tục một kế hoạch đã lỗi thời.

Planner cũng tạo action chain khi nhiều unit có thể đứng chung một tile. Thứ tự
unit trong action được dùng để chạy `PLANT → WATER`, `FEED +
COLLECT_FERTILIZER + CARE`, hoặc `WATER → HARVEST → PLANT → WATER` ngay trong cùng
turn. Đây là tối ưu cơ hội giữa các unit đã đứng sẵn cùng ô; planner không kéo
unit rời route để hội quân và không dùng độ rộng chain để tăng số hand. Các
deadline `WATER → HARVEST` và `FEED → COLLECT_FERTILIZER` đứng trước CARE.

Deadline lifecycle được áp dụng như sau:

- One-time crop trồng ngày `N` phải chạy `WATER → HARVEST` đúng ngày
  `N + Time to Max Yield`; sau đó mới `PLANT → WATER` cây thay thế.
- Ongoing crop phát `WATER → HARVEST` ngay trong ngày đầu tiên trạng thái
  `yield_units > 0`, không đợi đến max-yield day.
- Animal đặt ngày `N` phải `COLLECT_FERTILIZER` mỗi ngày kể từ `N + 1` khi
  `fertilizer_available`; thao tác này có priority 0 và đứng trước CARE.

Workload mỗi crop được ước lượng theo phase: 2 action khi khởi tạo, 1 action
trong ngày chỉ cần tưới, 4 action khi thu hoạch và trồng lại. Animal setup được
tính 4 service action (`PICKUP animal + BUILD + PLACE + FEED`) và một lần
`PICKUP WHEAT` dùng chung cho route. Animal đang sống tính đủ `FEED`, `HARVEST`,
`CARE` và `COLLECT_FERTILIZER` theo state hiện tại.

Trước khi rời shed, animal hand tạo manifest cho toàn route và thực hiện các
`PICKUP` liên tiếp: mọi con giống còn thiếu trước, sau đó lấy một lần đủ Wheat
cho các animal trong route. Khi crop one-time được thu hoạch, tile trống có mức
ưu tiên cao nhất; nếu không có unit nối tiếp thì chính crop owner lập tức
`PLANT`, rồi `WATER`, trước khi đi sang tile khác.

Chỉ trong ba ngày cuối mùa (day 27–29), sau khi hoàn tất mọi task priority 0/1
của route, worker đang mang nông sản quay về cửa shed gần nhất và `DROP` ngay.
Market planner nhìn thấy hàng ở observation kế tiếp và phát `SELL`. Các ngày
trước đó giữ lịch route bình thường. WATER/HARVEST/PLANT deadline vẫn đứng trước
chuyến về shed; Wheat đang được animal route mang đi FEED không bị nhận nhầm là
hàng cần bán.

Khi strategic policy thêm nhiều target vào layout, planner tự sinh thêm route
và market planner thuê thêm đúng số hand tương ứng.

## Kết quả smoke season

Với seed `42`, `episodeSteps=720`, `weedSpawnChance=0` và đối thủ `pass`, crop
policy chọn:

```text
day 4                 -> CARROT
day 6, NE 25 tiles    -> CARROT
day 7,9,10,17,19      -> MELON
day 20,24             -> WHEAT
day 27,28,29          -> không trồng lại các cohort đã hết window
```

Ngày 4, Melon có profit/day cao hơn nhưng không đủ tiền mua cho cả 9 tile nên
planner chọn Carrot. Planner mở `NE` ở ngày 5, mua và trồng đủ 25 Carrot ở ngày
6, rồi duy trì liên tục đủ `21 NW + 25 NE = 46` crop tile từ đầu ngày 7 đến hết
ngày 27 mà không có WEED. Các cohort NW trống cuối season là chủ ý khi không
còn bất kỳ loại crop nào kịp thu hoạch; planner không mua seed và không PLANT
trên các tile đó. NE vẫn đủ 25 tile đến ngày 29. Với DROP nhanh chỉ trong ba
ngày cuối, reward của replay hiện tại là `$89,307`.

- Farm hand tối đa trong một ngày: 13, chỉ ở các ngày setup/harvest nặng.
- Route được tách theo quadrant và dùng buffer 6 turn để không trồng cây mới ở
  hour 23 mà chưa kịp WATER.
- Full-season test audit tuổi của từng lệnh HARVEST và từng cặp
  `(day, animal tile)` bắt buộc COLLECT_FERTILIZER; smoke test còn kiểm tra đủ
  cả 46 crop tile trong production window mở rộng.
