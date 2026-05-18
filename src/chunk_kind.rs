/// Represents the semantic kind of a code chunk.
///
/// Using a dedicated enum eliminates string‑based bugs and makes merging logic
/// easier to reason about.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    Function,
    Impl,
    Struct,
    Enum,
    Trait,
    Mod,
    Macro,
    Const,
    Static,
    Type,
    Use,
    Gap,
    Mixed,
    Unknown,
}

impl std::fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ChunkKind::Function => "function",
            ChunkKind::Impl => "impl",
            ChunkKind::Struct => "struct",
            ChunkKind::Enum => "enum",
            ChunkKind::Trait => "trait",
            ChunkKind::Mod => "mod",
            ChunkKind::Macro => "macro",
            ChunkKind::Const => "const",
            ChunkKind::Static => "static",
            ChunkKind::Type => "type",
            ChunkKind::Use => "use",
            ChunkKind::Gap => "module_gap",
            ChunkKind::Mixed => "mixed_semantic_block",
            ChunkKind::Unknown => "unknown",
        };
        write!(f, "{}", s)
    }
}

impl From<&str> for ChunkKind {
    fn from(s: &str) -> Self {
        match s {
            "function" | "function_item" => ChunkKind::Function,
            "impl" | "impl_item" => ChunkKind::Impl,
            "struct" | "struct_item" => ChunkKind::Struct,
            "enum" | "enum_item" => ChunkKind::Enum,
            "trait" | "trait_item" => ChunkKind::Trait,
            "mod" | "mod_item" => ChunkKind::Mod,
            "macro_definition" => ChunkKind::Macro,
            "const" | "const_item" => ChunkKind::Const,
            "static" | "static_item" => ChunkKind::Static,
            "type" | "type_item" => ChunkKind::Type,
            "use" | "use_declaration" => ChunkKind::Use,
            "module_gap" => ChunkKind::Gap,
            "mixed_semantic_block" => ChunkKind::Mixed,
            _ => ChunkKind::Unknown,
        }
    }
}
