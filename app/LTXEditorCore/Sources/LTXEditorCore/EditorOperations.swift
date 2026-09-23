import Foundation

public enum EditorError: Error, Equatable, LocalizedError {
    case unknownAsset
    case unknownClip
    case unknownTake
    case wrongAssetKind
    case assetInUse
    case invalidRange
    case nothingToSplit

    public var errorDescription: String? {
        switch self {
        case .unknownAsset: return "That media is no longer in the project."
        case .unknownClip: return "That clip is no longer on the timeline."
        case .unknownTake: return "That take no longer exists."
        case .wrongAssetKind: return "This media can't go on that track."
        case .assetInUse: return "The media is used on the timeline. Remove its clips first."
        case .invalidRange: return "The clip would be shorter than one frame."
        case .nothingToSplit: return "Move the playhead inside a clip to split it."
        }
    }
}

// Timeline edits. Every operation validates first and leaves the project
// untouched when it throws, so callers can record undo snapshots before calling.
extension Project {

    // MARK: Assets

    /// Adds an asset, or returns the existing one with the same path.
    @discardableResult
    public mutating func addAsset(_ asset: Asset) -> Asset {
        if let existing = assets.first(where: { $0.path == asset.path }) {
            return existing
        }
        assets.append(asset)
        return asset
    }

    public mutating func removeAsset(_ id: UUID) throws {
        guard asset(id) != nil else { throw EditorError.unknownAsset }
        let usedByVideo = timeline.videoClips.contains { $0.takes.contains { $0.assetId == id } }
        let usedByAudio = timeline.audioClips.contains { $0.assetId == id }
        if usedByVideo || usedByAudio { throw EditorError.assetInUse }
        assets.removeAll { $0.id == id }
    }

    // MARK: Video track

    /// Inserts the whole asset as a new clip at `index` (default: end of the track).
    @discardableResult
    public mutating func insertVideoClip(assetId: UUID, at index: Int? = nil, label: String = "Original") throws -> UUID {
        guard let asset = asset(assetId) else { throw EditorError.unknownAsset }
        guard asset.kind == .video else { throw EditorError.wrongAssetKind }
        guard asset.duration >= minimumClipDuration else { throw EditorError.invalidRange }
        let clip = VideoClip(take: Take(assetId: assetId, label: label), inPoint: 0, outPoint: asset.duration)
        let position = min(max(index ?? timeline.videoClips.count, 0), timeline.videoClips.count)
        timeline.videoClips.insert(clip, at: position)
        return clip.id
    }

    /// Inserts at a timeline time: between clips at a boundary, else splits the clip under it.
    @discardableResult
    public mutating func insertVideoClip(assetId: UUID, atTime time: Double) throws -> UUID {
        guard asset(assetId)?.kind == .video else {
            throw asset(assetId) == nil ? EditorError.unknownAsset : EditorError.wrongAssetKind
        }
        guard let (index, offset) = timeline.videoClip(at: time) else {
            return try insertVideoClip(assetId: assetId)
        }
        if offset < minimumClipDuration / 2 {
            return try insertVideoClip(assetId: assetId, at: index)
        }
        var copy = self
        _ = try copy.splitVideoClip(atTime: time)
        let id = try copy.insertVideoClip(assetId: assetId, at: index + 1)
        self = copy
        return id
    }

    public mutating func moveVideoClip(_ id: UUID, to index: Int) throws {
        guard let from = timeline.videoClips.firstIndex(where: { $0.id == id }) else {
            throw EditorError.unknownClip
        }
        let clip = timeline.videoClips.remove(at: from)
        let target = min(max(index, 0), timeline.videoClips.count)
        timeline.videoClips.insert(clip, at: target)
    }

    /// Sets the clip's source range, clamped to its active take's media.
    public mutating func trimVideoClip(_ id: UUID, inPoint: Double, outPoint: Double) throws {
        guard let index = timeline.videoClips.firstIndex(where: { $0.id == id }) else {
            throw EditorError.unknownClip
        }
        let clip = timeline.videoClips[index]
        guard let media = asset(clip.activeTake.assetId) else { throw EditorError.unknownAsset }
        let newIn = min(max(inPoint, 0), media.duration)
        let newOut = min(max(outPoint, 0), media.duration)
        guard newOut - newIn >= minimumClipDuration - 1e-9 else { throw EditorError.invalidRange }
        timeline.videoClips[index].inPoint = newIn
        timeline.videoClips[index].outPoint = newOut
    }

    /// Splits the clip under `time` into two; returns the id of the second half.
    @discardableResult
    public mutating func splitVideoClip(atTime time: Double) throws -> UUID {
        guard let (index, offset) = timeline.videoClip(at: time) else { throw EditorError.nothingToSplit }
        let clip = timeline.videoClips[index]
        let cut = clip.inPoint + offset
        guard cut - clip.inPoint >= minimumClipDuration - 1e-9,
              clip.outPoint - cut >= minimumClipDuration - 1e-9 else {
            throw EditorError.nothingToSplit
        }
        var first = clip
        first.outPoint = cut
        var second = clip
        second.id = UUID()
        second.inPoint = cut
        timeline.videoClips[index] = first
        timeline.videoClips.insert(second, at: index + 1)
        return second.id
    }

    /// Removes the clip; later clips ripple left.
    public mutating func deleteVideoClip(_ id: UUID) throws {
        guard timeline.videoClips.contains(where: { $0.id == id }) else { throw EditorError.unknownClip }
        timeline.videoClips.removeAll { $0.id == id }
    }

    public mutating func setVideoClipVolume(_ id: UUID, _ volume: Double) throws {
        guard let index = timeline.videoClips.firstIndex(where: { $0.id == id }) else {
            throw EditorError.unknownClip
        }
        timeline.videoClips[index].volume = min(max(volume, 0), 1)
    }

    // MARK: Takes

    /// Adds `assetId` as a new take of the clip and (by default) makes it active.
    @discardableResult
    public mutating func addTake(toClip clipId: UUID, assetId: UUID, label: String, activate: Bool = true) throws -> UUID {
        guard let index = timeline.videoClips.firstIndex(where: { $0.id == clipId }) else {
            throw EditorError.unknownClip
        }
        guard let media = asset(assetId) else { throw EditorError.unknownAsset }
        guard media.kind == .video else { throw EditorError.wrongAssetKind }
        let take = Take(assetId: assetId, label: label)
        timeline.videoClips[index].takes.append(take)
        if activate {
            try activateTake(take.id, inClip: clipId)
        }
        return take.id
    }

    /// Switches the clip's active take, keeping its source range where the new media allows.
    public mutating func activateTake(_ takeId: UUID, inClip clipId: UUID) throws {
        guard let index = timeline.videoClips.firstIndex(where: { $0.id == clipId }) else {
            throw EditorError.unknownClip
        }
        var clip = timeline.videoClips[index]
        guard let take = clip.takes.first(where: { $0.id == takeId }) else { throw EditorError.unknownTake }
        guard let media = asset(take.assetId) else { throw EditorError.unknownAsset }
        clip.activeTakeId = takeId
        clip.outPoint = min(clip.outPoint, media.duration)
        clip.inPoint = min(clip.inPoint, max(0, clip.outPoint - minimumClipDuration))
        if clip.duration < minimumClipDuration {
            clip.inPoint = 0
            clip.outPoint = media.duration
        }
        timeline.videoClips[index] = clip
    }

    // MARK: Music track

    @discardableResult
    public mutating func addAudioClip(assetId: UUID, start: Double) throws -> UUID {
        guard let media = asset(assetId) else { throw EditorError.unknownAsset }
        guard media.kind == .audio || media.hasAudio else { throw EditorError.wrongAssetKind }
        let clip = AudioClip(assetId: assetId, start: max(0, start), inPoint: 0, outPoint: media.duration)
        timeline.audioClips.append(clip)
        return clip.id
    }

    public mutating func moveAudioClip(_ id: UUID, start: Double) throws {
        guard let index = timeline.audioClips.firstIndex(where: { $0.id == id }) else {
            throw EditorError.unknownClip
        }
        timeline.audioClips[index].start = max(0, start)
    }

    public mutating func trimAudioClip(_ id: UUID, inPoint: Double, outPoint: Double) throws {
        guard let index = timeline.audioClips.firstIndex(where: { $0.id == id }) else {
            throw EditorError.unknownClip
        }
        guard let media = asset(timeline.audioClips[index].assetId) else { throw EditorError.unknownAsset }
        let newIn = min(max(inPoint, 0), media.duration)
        let newOut = min(max(outPoint, 0), media.duration)
        guard newOut - newIn >= minimumClipDuration - 1e-9 else { throw EditorError.invalidRange }
        timeline.audioClips[index].inPoint = newIn
        timeline.audioClips[index].outPoint = newOut
    }

    public mutating func setAudioClipVolume(_ id: UUID, _ volume: Double) throws {
        guard let index = timeline.audioClips.firstIndex(where: { $0.id == id }) else {
            throw EditorError.unknownClip
        }
        timeline.audioClips[index].volume = min(max(volume, 0), 1)
    }

    public mutating func deleteAudioClip(_ id: UUID) throws {
        guard timeline.audioClips.contains(where: { $0.id == id }) else { throw EditorError.unknownClip }
        timeline.audioClips.removeAll { $0.id == id }
    }
}
