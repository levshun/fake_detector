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

<h3 align="left">Установка</h3>
<p align="left">

Данная библиотека была протестиирования для языка програмирования Python версии 3.11.

Процесс установки:

```pip install -i https://test.pypi.org/simple/ detect-ai```

```pip install -r .\requirements.txt```

Необходимые версии библиотек представлены в файле ```requirements.txt```.

Чтобы установить библиотеку, сначала скачайте репозиторий.

Затем создайте виртуальное окружение.

Далее воспользуйтесь командой ```pip``` для установки библиотек из файла ```requirements.txt```.

Итоговая структура проекта должна выглядеть следующим образом:
```
│   .gitignore
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
├───datasets
├───detect_ai
├───interface
├───models
├───tests
│   │   generating.py
│   │   modifying.py
│   │   swapping.py
│   │   swapping_performance.py
│   ├───generating_data
│   ├───swapping_data
└───tutorials
```

Каталоги ```models``` и ```datasets``` необходимо создать вручную.

Далее необходимо запустить проект в среде разработки для языка программирования Python, например, PyCharm, и использовать файлы из каталога ```tutorials``` для знакомства с библиотекой.
</p>

<h3 align="left">Модели</h3>
<p align="left">

Предобученные модели доступны по следующей ссылке: https://disk.yandex.ru/d/QVQL-HBabVkUoA.

После скачивания архива и его распаковки, структура файлов в каталоге ```models``` должна быть следующей: 
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
</p>

<h3 align="left">Наборы данных</h3>
<p align="left">

Наборы данных для тестирования доступны по следующей ссылке: https://disk.yandex.ru/d/9oX2v0c2E4zeFw.
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
