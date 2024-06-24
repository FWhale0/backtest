from __future__ import annotations


class FacBasic:
    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        subcategory: str,
        tags: list[str],
        data: dict[str, str],
    ):
        self.name = name
        self.description = description
        self.category = category
        self.subcategory = subcategory
        self.tags = tags
        self.data = data

    def __str__(self):
        return f"{self.name} ({self.category}/{self.subcategory})"

    def __repr__(self):
        return f"{self.name} ({self.category}/{self.subcategory})"

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.description == other.description
            and self.category == other.category
            and self.subcategory == other.subcategory
            and self.tags == other.tags
            and self.data == other.data
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(
            (
                self.name,
                self.description,
                self.category,
                self.subcategory,
                tuple(self.tags),
                tuple(self.data.items()),
            )
        )
