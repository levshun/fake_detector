[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/levshun/fake_detector/blob/master/README.en.md)

<h1 align="center">БИБЛИОТЕКА FAKE DETECTOR</h1>

<h3 align="left">Задачи</h3>
<p align="left">

Разработанная библиотека решает три основные задачи обнаружения:
- **Generating**: обнаружение сгенерированных изображений, содержащих изображение лица человека.
- **Modifying**: обнаружение измененных изображений лица человека.
- **Swapping**: обнаружение изображений лица человека с подменой лица.
</p>

<h3 align="left">Документация</h3>
<p align="left">

Онлайн-документация проекта представлена на сайте Read the Docs:  https://fake-detector.readthedocs.io.
</p>

<h3 align="left">Направления прикладного использования</h3>
<p align="left">

Разработанная библиотека может применяться по следующим направлениям:

1. <b>Экспертно-криминалистическая и судебная практика</b> — первичный анализ цифровых изображений и выявление признаков синтетической генерации/модификации.
2. <b>Правоохранительная деятельность</b> — анализ материалов, связанных с распространением поддельного визуального контента, инцидентами информационного воздействия, а также оценка достоверности материалов из открытых источников и каналов оперативного получения информации.
3. <b>Корпоративные системы репутационной защиты</b> — проверка изображений, используемых в публичных коммуникациях, поддержка расследования репутационных атак и снижение рисков публикации/распространения недостоверных визуальных материалов.
4. <b>Системы модерации контента в социальных сетях</b> — вспомогательный инструмент для информационных систем с пользовательским контентом (например, выявление подозрительных объектов для последующей проверки модераторами/экспертами).
5. <b>Образовательная деятельность</b> — подготовка и проведение учебных курсов, лабораторных работ и практикумов по цифровой криминалистике, анализу цифровых следов, а также по тематике доверенного и безопасного применения технологий искусственного интеллекта.
6. <b>Научно-исследовательская деятельность</b> — использование библиотеки и наборов данных как открытой базы для воспроизводимых экспериментов, сравнения алгоритмов и формирования методических материалов.

</p>

<h3 align="left">Минимальные технические требования</h3>
<p align="left">
Минимальная рекомендуемая конфигурация вычислительной платформы: CPU - современный многоядерный процессор уровня не ниже 6-8 физических ядер (Intel Core i5/i7 или AMD Ryzen 5/7 сопоставимого поколения), либо Apple Silicon уровня Apple M4 и выше, с тактовой частотой порядка 3.0 GHz и выше (для x86-платформ); RAM - не менее 16 GB; GPU - опционально.

Для обеспечения требуемой производительности обработки (время обработки одного изображения для одного типа подделки - не более 5 секунд) рекомендуется конфигурация не хуже: CPU Intel Core i9-12900KF (3.20 GHz) или эквивалентный по производительности, RAM 32 GB (DDR5), GPU NVIDIA GeForce RTX 3090 Ti (рекомендуется 24 GB VRAM). При использовании более слабой инфраструктуры (процессор более низкого класса, меньший объем ОЗУ, отсутствие GPU или GPU с меньшим объемом видеопамяти) время обработки может увеличиваться и в отдельных режимах превышать 5 секунд.

Для запуска и штатного использования открытой библиотеки могут использоваться следующие операционные системы: Windows 11, Debian 12 и macOS 26.
</p>

<h3 align="left">Установка</h3>
<p align="left">

Данная библиотека была протестирована для языка програмирования Python версии 3.11, 3.12.

Процесс установки:

Шаг 1. Создайте проект в Python IDE, скопировав его из VCS по ссылке ```https://github.com/levshun/fake_detector/```. Также можно сделать это из консоли, запустив команду:

```git clone https://github.com/levshun/fake_detector/```

Структура файлов и папок должна выглядеть следующим образом:

```
│   .gitignore
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
├───detect_ai
├───interface
├───tests
│   │   generating.py
│   │   modifying.py
│   │   swapping.py
│   │   swapping_performance.py
│   ├───generating_data
│   ├───swapping_data
└───tutorials
```

Шаг 2. Настройте интерпретатор в Python IDE, взяв за основу Python 3.11-3.12. Используйте виртуальную среду ```.venv``` для установки зависимостей.

Шаг 3. Необходимые версии сторонних библиотек (зависимости) представлены в файле ```requirements.txt```. Если библиотеки не были установлены при создании проекта, это можно сделать вручную с помощью следующей команды:

```pip install -r requirements.txt```

Шаг 5. Установите библиотеку используя TEST PyPi.

```pip install -i https://test.pypi.org/simple/ detect-ai```

Шаг 6. Выгрузите архив с предобученными моделям по ссылке: [```https://disk.yandex.ru/d/QVQL-HBabVkUoA```](https://disk.yandex.ru/d/08_in5a5dZ0K4A). Скопируйте архив в корень проекта и распакуйте его. В корне проекта должна появиться структура каталогов и файлов следующего вида:

```
models
├───generating
│       convnext_model.pth
│       eva_model.pth
│       final_decisiontree.pkl
│
├───modifying
│   │   face_landmarker_v2_with_blendshapes.task
│   │
│   ├───binary
│   │   ├───bald_gan
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   ├───beauty_gan
│   │   │   ├───eff_net_b3
│   │   │   │       class.json
│   │   │   │       eff_net_b3.keras
│   │   │   │
│   │   │   └───inception_v3
│   │   │           class.json
│   │   │           inception_v3.keras
│   │   │
│   │   ├───b_lfw
│   │   │   └───effnetv2s
│   │   │           class.json
│   │   │           effnetv2s.keras
│   │   │
│   │   ├───makeup_wild
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   ├───pilgram
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   └───qwen
│   │       └───eff_net_b3
│   │               class.json
│   │               eff_net_b3.keras
│   │
│   └───multiclass
│       ├───pilgram
│       │   └───eff_net_b3
│       │           class.json
│       │           eff_net_b3.keras
│       │
│       └───tool
│           └───eff_net_b3
│                   class.json
│                   eff_net_b3.keras
│
└───swapping
    │   face_detection_yunet_2023mar.onnx
    │   shape_predictor_68_face_landmarks.dat
    │
    ├───efficientnet
    │       effnet_github.pth
    │       effnet_rgb.pth
    │       effnet_roop.pth
    │       effnet_segmind.pth
    │
    └───feature_based
            catboost_lbp.cbm
            catboost_tf_sf.cbm
            meta_model_ensemble.pkl
            random_forest_ef.pkl
            random_forest_fl.pkl
```

Шаг 7. Выгрузите архив с данными для тестирования по ссылке: [https://disk.yandex.ru/d/YHuOkp-tSEX_Kg](https://disk.yandex.ru/d/YHuOkp-tSEX_Kg). Скопируйте архив в корень проекта и распакуйте его. В корне проекта должна появиться структура каталогов и файлов следующего вида:

```
datasets
├── generating
└── modifying
    ├── bald_gan
    │   ├── modification
    │   └── original
    ├── beauty_gan
    │   ├── modification
    │   └── original
    ├── pilgram
    │   ├── modification
    │   ├── modification_multi
    │   │   ├── blending
    │   │   ├── css
    │   │   └── instagram
    │   └── original
    └── qwen
        ├── modification
        └── original
```

Шаг 8. Используйте интерактивные ноутбуки из каталога ```tutorials``` для знакомства с библиотекой.
</p>

<h3 align="left">Модели</h3>
<p align="left">
Архив с предобученными моделями доступен по следующей ссылке: [https://disk.yandex.ru/d/QVQL-HBabVkUoA](https://disk.yandex.ru/d/08_in5a5dZ0K4A). 
</p>

<h3 align="left">Наборы данных</h3>
<p align="left">
Архив с наборами данных для тестирования доступен по следующей ссылке: [https://disk.yandex.ru/d/9oX2v0c2E4zeFw](https://disk.yandex.ru/d/08_in5a5dZ0K4A).
</p>

<h3 align="left">Зеркало</h3>
<p align="left">

Зеркало репозитория доступно на GitLab: https://gitlab.com/levshun/fake_detector.
</p>

<h3 align="left">Контакты</h3>
<p align="left">

Вы можете связаться с нами, написав на следующий почтовый адрес: ```dmitry.levshun@gmail.com```.
</p>

<h3 align="left">Благодарности</h3>
<p align="left">

Разработка данной библиотеки ведется при поддержке Фонда содействия инновациям по договору 50ГУКодИИС13-D7/94529 от 03.06.2024.
</p>
