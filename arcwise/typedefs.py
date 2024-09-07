from typing import Any


from pydantic import BaseModel


class ForeignKey(BaseModel):
    columns: list[str]
    reference_table: str
    reference_columns: list[str]
    relationship: str | None


class ColumnInfo(BaseModel):
    name: str
    original_name: str | None = None
    type: str
    ai_description: str | None = None
    description: str | None = None
    value_description: str | None = None
    null_fraction: float
    unique_count: int
    unique_fraction: float
    sample_values: list[Any]
    min_value: Any
    max_value: Any


class Table(BaseModel):
    name: str
    ai_description: str | None = None
    row_count: int
    primary_key: list[str]
    foreign_keys: list[ForeignKey] = []
    columns: list[ColumnInfo]

    def format_for_column_prediction(self) -> str:
        lines = [
            f"# Table: {self.name}",
            *([f"Description: {self.ai_description}"] if self.ai_description else []),
            f"Rows: {self.row_count}",
        ]

        if self.primary_key:
            lines.append(f"Primary key: ({', '.join(self.primary_key)})")

        for fkey in self.foreign_keys:
            line = (
                f"Foreign key: ({', '.join(fkey.columns)})"
                f" references {fkey.reference_table} "
                f"({', '.join(fkey.reference_columns)})"
            )
            if fkey.relationship:
                line += f" {fkey.relationship}"
            lines.append(line)

        lines.append("Columns:")
        for col in self.columns:
            line = f"{self.name}.{col.name}\t{col.type.upper()}"
            if col.ai_description:
                description_clean = col.ai_description.replace("\n", " ")[:512]
                line += f"\t{description_clean}"
            lines.append(line)

        return "\n".join(lines)


class Database(BaseModel):
    name: str
    tables: list[Table]


class SchemaPredictions(BaseModel):
    class OutputType(BaseModel):
        type: str
        description: str

    class InputColumn(BaseModel):
        column: str
        description: str | None = None
        votes: int | None = None

    output_types: list[OutputType]
    input_columns: list[InputColumn]
    raw_prediction: str | None = None


class BIRDQuestion(BaseModel):
    db_id: str
    question: str
    evidence: str | None = None
    SQL: str | None = None
    question_id: int | None = None
    schema_predictions: SchemaPredictions | None = None
    filtered_schema: str | None = None

    def question_evidence(self) -> str:
        res = self.question.strip()
        if self.evidence:
            res += f"\nContext: {self.evidence.strip()}"
        return res

    def question_hint(self) -> str:
        res = self.question.strip()
        if self.evidence:
            res += f"\nHint: {self.evidence.strip()}"
        return res
