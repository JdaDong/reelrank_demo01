-- ReelRank 数仓 DDL（幂等：ETL 使用 CREATE OR REPLACE 重建表）

CREATE TABLE IF NOT EXISTS dim_movie (
    movie_id          BIGINT PRIMARY KEY,
    title             VARCHAR,
    original_title    VARCHAR,
    overview          VARCHAR,
    tagline           VARCHAR,
    release_date      DATE,
    year              INTEGER,
    runtime           INTEGER,
    budget            BIGINT,
    revenue           BIGINT,
    popularity        DOUBLE,
    vote_average      DOUBLE,
    vote_count        BIGINT,
    original_language VARCHAR,
    adult             BOOLEAN,
    status            VARCHAR,
    imdb_id           VARCHAR,
    poster_path       VARCHAR,
    backdrop_path     VARCHAR,
    genre_names       VARCHAR[],
    keyword_names     VARCHAR[]
);

CREATE TABLE IF NOT EXISTS bridge_movie_genre (
    movie_id   BIGINT,
    genre_id   BIGINT,
    genre_name VARCHAR
);

CREATE TABLE IF NOT EXISTS bridge_movie_keyword (
    movie_id     BIGINT,
    keyword_id   BIGINT,
    keyword_name VARCHAR
);

CREATE TABLE IF NOT EXISTS bridge_movie_cast (
    movie_id   BIGINT,
    person_id  BIGINT,
    person_name VARCHAR,
    cast_order INTEGER,
    character  VARCHAR
);

CREATE TABLE IF NOT EXISTS bridge_movie_director (
    movie_id    BIGINT,
    person_id   BIGINT,
    person_name VARCHAR
);

CREATE TABLE IF NOT EXISTS bridge_movie_company (
    movie_id     BIGINT,
    company_id   BIGINT,
    company_name VARCHAR
);

-- 关系边：similar=TMDB 相似影片，recommendation=TMDB 推荐影片
CREATE TABLE IF NOT EXISTS item_sim_edge (
    src_movie_id BIGINT,
    dst_movie_id BIGINT,
    relation     VARCHAR,
    rank_pos     INTEGER,
    PRIMARY KEY (src_movie_id, dst_movie_id, relation)
);

CREATE TABLE IF NOT EXISTS fact_user_event (
    user_id    BIGINT,
    movie_id   BIGINT,
    event_type VARCHAR,
    rating     DOUBLE,
    event_time TIMESTAMP,
    source     VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_ad (
    ad_id           BIGINT PRIMARY KEY,
    advertiser_id   BIGINT,
    advertiser_name VARCHAR,
    movie_id        BIGINT,
    title           VARCHAR,
    bid             DOUBLE,
    daily_budget    DOUBLE,
    spent           DOUBLE,
    freq_cap        INTEGER,
    targeting       VARCHAR,
    status          VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_ad_delivery (
    request_id VARCHAR,
    ad_id      BIGINT,
    slot       INTEGER,
    pctr       DOUBLE,
    ecpm       DOUBLE,
    price      DOUBLE,
    event_time TIMESTAMP
);
