# database_tree_text.py
import os
from sqlalchemy import create_engine, inspect

DATABASE_PATH = r"C:\Users\pc\PycharmProjects\pythonProject\problem_management_ststem\instance\problem_management.db"

if not os.path.exists(DATABASE_PATH):
    raise FileNotFoundError(f"The database file does not exist at: {DATABASE_PATH}")

DATABASE_URI = f"sqlite:///{os.path.abspath(DATABASE_PATH)}"
engine = create_engine(DATABASE_URI)

try:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
except Exception as e:
    print(f"Error inspecting database: {e}")
    exit(1)

print("Database Schema:")
for table_name in tables:
    print(f"└── TABLE: {table_name}")
    columns = inspector.get_columns(table_name)
    foreign_keys = inspector.get_foreign_keys(table_name)

    for i, column in enumerate(columns):
        col_name = column['name']
        col_type = str(column['type'])  # تحويل النوع إلى نص
        col_nullable = column['nullable']
        col_default = column.get('default', '')  # .get لتجنب خطأ إذا لم يكن المفتاح موجودًا
        col_pk = column.get('primary_key', 0) == 1  # التحقق مما إذا كان مفتاحًا أساسيًا

        prefix = "    ├──" if i < len(columns) - 1 else "    └──"

        details = f"Type: {col_type}, Nullable: {col_nullable}"
        if col_pk:
            details += ", PK"
        if col_default is not None and str(col_default).strip() != '':
            details += f", Default: {col_default}"

        # التحقق من المفاتيح الخارجية
        fk_info = ""
        for fk in foreign_keys:
            if col_name in fk['constrained_columns']:
                fk_info = f" (FK -> {fk['referred_table']}.{fk['referred_columns'][0]})"
                break

        print(f"{prefix} COLUMN: {col_name}{fk_info} ({details})")
    print("")  # سطر فارغ للفصل بين الجداول