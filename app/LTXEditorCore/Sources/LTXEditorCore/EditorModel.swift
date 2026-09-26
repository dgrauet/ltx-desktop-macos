import Foundation

// Persistent editor document: a project's asset bin and timeline.
// Times are seconds (Double) on the timeline or within an asset; FCPXML export
// and composition building convert to frames / CMTime at the edges.

public enum AssetKind: String, Codable, Hashable, Sendable {
    case video
    case audio
}

/// A media file referenced by absolute path (never copied into the project).
public struct Asset: Codable, Identifiable, Hashable, Sendable {
    public var id: UUID
    public var path: String
    public var kind: AssetKind
    public var name: String
    /// Media duration in seconds.
    public var duration: Double
    public var hasAudio: Bool
    public var width: Int?
    public var height: Int?
    /// Backend job id when the asset came from a generation.
    public var sourceJobId: String?

    public init(
        id: UUID = UUID(), path: String, kind: AssetKind, name: String, duration: Double,
        hasAudio: Bool, width: Int? = nil, height: Int? = nil, sourceJobId: String? = nil
    ) {
        self.id = id
        self.path = path
        self.kind = kind
        self.name = name
        self.duration = duration
        self.hasAudio = hasAudio
        self.width = width
        self.height = height
        self.sourceJobId = sourceJobId
    }
}

/// One alternative piece of media for a video clip slot (original, retake, extension…).
public struct Take: Codable, Identifiable, Hashable, Sendable {
    public var id: UUID
    public var assetId: UUID
    public var label: String

    public init(id: UUID = UUID(), assetId: UUID, label: String) {
        self.id = id
        self.assetId = assetId
        self.label = label
    }
}

/// A clip on the (magnetic) video track. Plays `[inPoint, outPoint)` of its active take.
public struct VideoClip: Codable, Identifiable, Hashable, Sendable {
    public var id: UUID
    public var takes: [Take]
    public var activeTakeId: UUID
    public var inPoint: Double
    public var outPoint: Double
    /// Linear gain for the clip's embedded audio, 0...1.
    public var volume: Double

    public init(id: UUID = UUID(), take: Take, inPoint: Double, outPoint: Double, volume: Double = 1) {
        self.id = id
        self.takes = [take]
        self.activeTakeId = take.id
        self.inPoint = inPoint
        self.outPoint = outPoint
        self.volume = volume
    }

    public var duration: Double { outPoint - inPoint }

    public var activeTake: Take {
        takes.first { $0.id == activeTakeId } ?? takes[0]
    }
}

/// A clip on the music track, placed at an absolute timeline position.
public struct AudioClip: Codable, Identifiable, Hashable, Sendable {
    public var id: UUID
    public var assetId: UUID
    /// Timeline position of the clip's first sample, seconds.
    public var start: Double
    public var inPoint: Double
    public var outPoint: Double
    public var volume: Double

    public init(
        id: UUID = UUID(), assetId: UUID, start: Double, inPoint: Double, outPoint: Double,
        volume: Double = 1
    ) {
        self.id = id
        self.assetId = assetId
        self.start = start
        self.inPoint = inPoint
        self.outPoint = outPoint
        self.volume = volume
    }

    public var duration: Double { outPoint - inPoint }
    public var end: Double { start + duration }
}

public struct Timeline: Codable, Hashable, Sendable {
    public var videoClips: [VideoClip]
    public var audioClips: [AudioClip]
    /// Silences the video clips' embedded audio (music track unaffected).
    public var videoTrackMuted: Bool

    public init(videoClips: [VideoClip] = [], audioClips: [AudioClip] = [], videoTrackMuted: Bool = false) {
        self.videoClips = videoClips
        self.audioClips = audioClips
        self.videoTrackMuted = videoTrackMuted
    }

    /// Length of the edit: the video track, or a music clip running past it.
    public var duration: Double {
        let video = videoClips.reduce(0) { $0 + $1.duration }
        let audio = audioClips.map(\.end).max() ?? 0
        return max(video, audio)
    }

    /// Timeline start time of every video clip, in order.
    public var videoClipStarts: [Double] {
        var t = 0.0
        return videoClips.map { clip in
            defer { t += clip.duration }
            return t
        }
    }

    public func startTime(ofVideoClip id: UUID) -> Double? {
        guard let index = videoClips.firstIndex(where: { $0.id == id }) else { return nil }
        return videoClipStarts[index]
    }

    /// The video clip under `time` and the offset into it (clip-local, from its start).
    public func videoClip(at time: Double) -> (index: Int, offset: Double)? {
        for (index, start) in videoClipStarts.enumerated() {
            let clip = videoClips[index]
            if time >= start && time < start + clip.duration {
                return (index, time - start)
            }
        }
        return nil
    }
}

public struct Project: Codable, Identifiable, Hashable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var id: UUID
    public var name: String
    public var createdAt: Date
    public var modifiedAt: Date
    public var fps: Int
    public var assets: [Asset]
    public var timeline: Timeline

    public init(id: UUID = UUID(), name: String, fps: Int = 24, now: Date = Date()) {
        self.schemaVersion = Project.currentSchemaVersion
        self.id = id
        self.name = name
        self.createdAt = now
        self.modifiedAt = now
        self.fps = fps
        self.assets = []
        self.timeline = Timeline()
    }

    public func asset(_ id: UUID) -> Asset? {
        assets.first { $0.id == id }
    }

    /// Shortest clip length the editor allows: one frame.
    public var minimumClipDuration: Double { 1.0 / Double(fps) }
}
