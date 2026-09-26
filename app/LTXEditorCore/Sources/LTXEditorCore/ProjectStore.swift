import Foundation

/// Lightweight listing entry, read without keeping whole projects around.
public struct ProjectSummary: Identifiable, Hashable, Sendable {
    public let id: UUID
    public let name: String
    public let modifiedAt: Date
    public let clipCount: Int
    public let duration: Double
}

/// File-backed project persistence: `<root>/<uuid>/project.json`, written atomically.
public struct ProjectStore: Sendable {
    public static var defaultRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ltx-desktop/projects", isDirectory: true)
    }

    public let root: URL

    public init(root: URL = ProjectStore.defaultRoot) {
        self.root = root
    }

    private func fileURL(_ id: UUID) -> URL {
        root.appendingPathComponent(id.uuidString, isDirectory: true)
            .appendingPathComponent("project.json")
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    public func save(_ project: Project) throws {
        let url = fileURL(project.id)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let data = try Self.encoder.encode(project)
        try data.write(to: url, options: .atomic)
    }

    public func load(_ id: UUID) throws -> Project {
        try Self.decoder.decode(Project.self, from: Data(contentsOf: fileURL(id)))
    }

    /// All readable projects, most recently modified first. Unreadable ones are skipped.
    public func list() -> [ProjectSummary] {
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil
        )) ?? []
        return dirs.compactMap { dir -> ProjectSummary? in
            guard let id = UUID(uuidString: dir.lastPathComponent),
                  let project = try? load(id) else { return nil }
            return ProjectSummary(
                id: project.id, name: project.name, modifiedAt: project.modifiedAt,
                clipCount: project.timeline.videoClips.count, duration: project.timeline.duration
            )
        }
        .sorted { $0.modifiedAt > $1.modifiedAt }
    }

    public func delete(_ id: UUID) throws {
        let dir = root.appendingPathComponent(id.uuidString, isDirectory: true)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }
}
