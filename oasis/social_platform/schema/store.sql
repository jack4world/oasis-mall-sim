CREATE TABLE store (
    store_id INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    floor INTEGER DEFAULT 1,
    brand_tier TEXT DEFAULT 'mid',
    avg_spend_per_visit REAL DEFAULT 0,
    monthly_rent REAL DEFAULT 0,
    area_sqm REAL DEFAULT 0
);
