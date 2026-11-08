# Movie Model Backend

HTTP API системы рекомендаций фильмов. Реализованы профили, локальный каталог фильмов и сериалов, импорт метаданных из TMDB, хранение оценок и избранного по профилям в SQLite и миграции. Рекомендации объединяют оценочную модель MovieLens и подбор по жанрам и описаниям TMDB; популярная подборка служит запасным вариантом.

Стек: Python 3.12+, FastAPI, SQLAlchemy async, Alembic, Dishka. Предметная область и сценарии приложения отделены от HTTP и базы данных. Настройки находятся в `src/config`, сущности — в `src/domain/entities`, HTTP DTO — в `src/presentation/http/dto`.

## Установка

Выполняйте команды из корня репозитория.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.lock -e ".[dev]"
cp .env.example .env
```

`requirements.lock` фиксирует версии зависимостей, включая инструменты разработки. Файл `.env` не включается в Git. Параметры окружения имеют префикс `MOVIE_`; пример конфигурации находится в `.env.example`.

## База данных и запуск

```sh
alembic upgrade head
uvicorn main:create_app --factory --host 127.0.0.1 --port 8000 --reload
```

SQLite по умолчанию хранится в `data/movie_model.sqlite3`. Путь в `MOVIE_DATABASE_URL` относительно текущей рабочей папки; для независимого пути используйте абсолютный URL SQLite. Каталог для БД создаётся при инициализации подключения. Схема изменяется только через Alembic: приложение не создаёт таблицы при запуске.

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI: http://127.0.0.1:8000/openapi.json
- Процесс: http://127.0.0.1:8000/health
- Готовность БД и схемы: http://127.0.0.1:8000/ready

Первый этап предназначен для локальной разработки. Авторизации ещё нет: список профилей общий, знание UUID не подтверждает право доступа. До внешнего размещения необходимо добавить пользователей и проверку доступа к профилям. CORS ограничивает браузерные запросы и не заменяет авторизацию.

## Пример работы с профилями

```sh
curl -X POST http://127.0.0.1:8000/api/v1/profiles \
  -H 'Content-Type: application/json' \
  -d '{"name":"Арсений"}'

curl http://127.0.0.1:8000/api/v1/profiles
```

Создание возвращает HTTP 201, UUID профиля, имя и время создания в UTC. `Location` содержит адрес созданного профиля. Имя очищается от пробелов по краям и должно содержать 1–80 символов. Одинаковые отображаемые имена допустимы; профили различаются по UUID.

## Поиск и сохранение фильмов

Интерфейс использует `GET /api/v1/discovery/movies`: популярные подборки и поиск TMDB через наш бэкенд. Для просмотра карточки используется `GET /api/v1/discovery/movies/{media_type}/{tmdb_id}`. Эти методы не обращаются к репозиторию фильмов и не сохраняют результаты в БД.

Укажите `MOVIE_TMDB_API_KEY` в `.env`. Таймаут HTTP задаёт `MOVIE_TMDB_TIMEOUT` (по умолчанию 8 секунд). Кэш страниц и карточек действует `MOVIE_TMDB_CACHE_TTL` секунд (по умолчанию 300), размер каждой из двух групп ограничен `MOVIE_TMDB_CACHE_ENTRIES` (по умолчанию 128). Ошибки не кэшируются. Одинаковые одновременные запросы используют одну загрузку. Справочник жанров кэшируется отдельно на сутки.

Когда пользователь добавляет фильм в избранное или ставит оценку, фронтенд отправляет тип и TMDB ID либо UUID сохранённой карточки. Бэкенд сам разрешает ссылку, загружает только отсутствующий фильм и сохраняет карточку вместе с действием в одной транзакции. Оценки 1–5 и избранное привязаны к UUID профиля; чтение коллекции и удаление действий не требуют TMDB.

`GET /api/v1/profiles/{profile_id}/library` возвращает полный снимок коллекции. PUT/DELETE `/favorites/{movie_id}` и `/ratings/{movie_id}` изменяют сохранённые записи; PUT `/favorites` и `/ratings` принимают внешний ID. Явный POST `/library/import` переносит до 20 записей прежней браузерной коллекции за транзакцию, сохраняя уже существующие серверные оценки.

`GET /api/v1/movies` остаётся локальным каталогом сохранённых записей, необходимым для пользовательских ссылок и ML. Этот метод не является поиском по всему TMDB. POST импорта страницы остаётся служебной операцией для наполнения датасетов и не используется обычным интерфейсом.

Подробнее: [HTTP API](docs/api.md), [архитектура](docs/architecture.md).

## Обучение и рекомендации

Датасет MovieLens 32M хранится отдельно от рабочей БД. Подготовка выполняется потоково, сходство фильмов рассчитывается блоками. Установка ML-зависимостей и воспроизводимый запуск:

```sh
python -m pip install -c requirements.lock -e ".[dev,ml]"
python -m ml download --source 32m
python -m ml prepare --source 32m
python -m ml train --output artifacts/recommendations/movielens-32m-top30-v1 --evaluation-users 2000
```

ML-артефакт необязателен: без него сервис использует подбор по содержанию и популярности TMDB. Для подключения совместимого артефакта укажите `MOVIE_RECOMMENDATION_MODEL_PATH=artifacts/recommendations/movielens-32m-top30-v1/model` в `.env` и перезапустите сервис. `GET /api/v1/profiles/{profile_id}/recommendations?limit=10` вернёт фильмы по оценкам профиля, без уже оценённых. Для профиля без подходящей истории используется явно обозначенная популярная подборка. Новые фильмы могут попасть в выдачу через TMDB независимо от каталога ML-модели. Сериалы не участвуют в ML.

Результаты сравнения алгоритмов и артефакты сохраняются в `artifacts/`, а не в рабочую БД. Подробности: [гибридный сервис](docs/recommendations.md), [обучение и протокол оценки](docs/machine-learning.md).

## Исследование данных

Полная статистика подготовленного MovieLens и графики рассчитываются отдельно от обучения:

```sh
python -m pip install -c requirements.lock -e ".[ml,analysis]"
python -m ml analyze --output artifacts/analysis/movielens-32m-eda-v1
```

Команда сохраняет JSON, CSV и PNG/SVG в artifacts, не меняя рабочую БД и модель. Методика и описание файлов: [исследование датасета](docs/dataset-analysis.md).

## Расширенная оценка

Восстановление тестовой копии с зафиксированными параметрами, парные доверительные интервалы и анализ групп:

```sh
python -m ml assess \
  --training-run artifacts/recommendations/movielens-32m-top30-v1 \
  --output artifacts/evaluation/movielens-32m-top30-quality-v1
```

Требуются extra ml и analysis. Рабочий артефакт не изменяется. Методика: [расширенная оценка модели](docs/model-evaluation.md).

## Разработка

```sh
pytest
ruff check .
ruff format --check .
alembic check
```

Тесты используют отдельные временные БД и имитацию TMDB, не меняя рабочие данные и не обращаясь к внешнему API. Команда `alembic check` проверяет соответствие текущей схемы ORM-моделям; перед ней примените миграции.

Для изменения схемы:

```sh
alembic revision --autogenerate -m "describe schema change"
# Проверьте созданную миграцию перед применением
alembic upgrade head
```

## Документация

- [Архитектура](docs/architecture.md)
- [HTTP API](docs/api.md)
- [Машинное обучение](docs/machine-learning.md)
- [Исследование датасета](docs/dataset-analysis.md)
- [Расширенная оценка](docs/model-evaluation.md)

## Репозитории

- [Бэкенд](https://github.com/Arsgaz/movie_model_backend)
- [Фронтенд](https://github.com/Arsgaz/movie_model_frontend), локально в `../movie_model_frontend`.

## Контейнерный запуск

При соседнем каталоге movie_model_frontend и настроенном .env выполните `docker compose up --build -d --wait`. Интерфейс доступен на http://localhost:8080, API — на http://localhost:8081. Для подключения готовой модели: `docker compose -f compose.yaml -f compose.model.yaml up --build -d --wait`. БД сохраняется отдельно от образов. Подробнее: [развёртывание](docs/deployment.md), [ER-диаграмма](docs/data-model.md), [схемы компонентов](docs/system-design.md).

Характеристики текущего артефакта, измеренные метрики и ограничения применения: [карточка модели](docs/model-card.md).
