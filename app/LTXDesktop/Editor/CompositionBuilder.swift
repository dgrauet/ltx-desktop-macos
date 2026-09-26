import AVFoundation
import CoreGraphics

/// Builds a playable/exportable AVFoundation composition from an editor project.
///
/// Video clips are laid back to back on one video track (their embedded audio on
/// a parallel track). Each music clip gets its own audio track: `insertTimeRange`
/// shifts existing content, so overlapping clips can't share one track.
/// Clips of a different size are scaled to fit the first clip's render size.
enum CompositionBuilder {
    struct Result {
        let composition: AVMutableComposition
        let videoComposition: AVMutableVideoComposition?
        let audioMix: AVMutableAudioMix
        let renderSize: CGSize
    }

    enum BuildError: LocalizedError {
        case missingMedia(String)

        var errorDescription: String? {
            switch self {
            case .missingMedia(let path): return "Media file not found: \(path)"
            }
        }
    }

    static func time(_ seconds: Double) -> CMTime {
        CMTime(seconds: seconds, preferredTimescale: 600)
    }

    static func build(_ project: Project) async throws -> Result {
        let composition = AVMutableComposition()
        let videoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
        let clipAudioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)

        var renderSize: CGSize?
        var layerInstructions: [AVMutableVideoCompositionInstruction] = []
        let clipAudioParams = AVMutableAudioMixInputParameters(track: clipAudioTrack)
        var musicParams: [AVMutableAudioMixInputParameters] = []

        var cursor = CMTime.zero
        for clip in project.timeline.videoClips {
            guard let asset = project.asset(clip.activeTake.assetId) else { continue }
            guard FileManager.default.fileExists(atPath: asset.path) else {
                throw BuildError.missingMedia(asset.path)
            }
            let media = AVURLAsset(url: URL(fileURLWithPath: asset.path))
            let range = CMTimeRange(start: time(clip.inPoint), duration: time(clip.duration))

            if let source = try await media.loadTracks(withMediaType: .video).first, let videoTrack {
                try videoTrack.insertTimeRange(range, of: source, at: cursor)
                let natural = try await source.load(.naturalSize)
                let transform = try await source.load(.preferredTransform)
                let size = natural.applying(transform)
                let clipSize = CGSize(width: abs(size.width), height: abs(size.height))
                let target = renderSize ?? clipSize
                renderSize = target

                let instruction = AVMutableVideoCompositionInstruction()
                instruction.timeRange = CMTimeRange(start: cursor, duration: range.duration)
                let layer = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
                let scale = min(target.width / clipSize.width, target.height / clipSize.height)
                let dx = (target.width - clipSize.width * scale) / 2
                let dy = (target.height - clipSize.height * scale) / 2
                layer.setTransform(
                    transform.concatenating(CGAffineTransform(scaleX: scale, y: scale))
                        .concatenating(CGAffineTransform(translationX: dx, y: dy)),
                    at: cursor
                )
                instruction.layerInstructions = [layer]
                layerInstructions.append(instruction)
            }
            if let source = try await media.loadTracks(withMediaType: .audio).first, let clipAudioTrack {
                try clipAudioTrack.insertTimeRange(range, of: source, at: cursor)
                let gain = project.timeline.videoTrackMuted ? 0 : Float(clip.volume)
                clipAudioParams.setVolume(gain, at: cursor)
            }
            cursor = cursor + range.duration
        }

        for clip in project.timeline.audioClips {
            guard let asset = project.asset(clip.assetId) else { continue }
            guard FileManager.default.fileExists(atPath: asset.path) else {
                throw BuildError.missingMedia(asset.path)
            }
            let media = AVURLAsset(url: URL(fileURLWithPath: asset.path))
            guard let source = try await media.loadTracks(withMediaType: .audio).first,
                  let musicTrack = composition.addMutableTrack(
                      withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid
                  ) else { continue }
            let range = CMTimeRange(start: time(clip.inPoint), duration: time(clip.duration))
            try musicTrack.insertTimeRange(range, of: source, at: time(clip.start))
            let params = AVMutableAudioMixInputParameters(track: musicTrack)
            params.setVolume(Float(clip.volume), at: .zero)
            musicParams.append(params)
        }

        let audioMix = AVMutableAudioMix()
        audioMix.inputParameters = [clipAudioParams] + musicParams

        // Video-composition instructions must cover the whole composition without gaps:
        // when music runs past the last video clip, render black for the tail.
        if composition.duration > cursor {
            let tail = AVMutableVideoCompositionInstruction()
            tail.timeRange = CMTimeRange(start: cursor, end: composition.duration)
            tail.backgroundColor = CGColor(gray: 0, alpha: 1)
            layerInstructions.append(tail)
        }

        var videoComposition: AVMutableVideoComposition?
        if let renderSize, !layerInstructions.isEmpty {
            let vc = AVMutableVideoComposition()
            vc.renderSize = renderSize
            vc.frameDuration = CMTime(value: 1, timescale: CMTimeScale(max(project.fps, 1)))
            vc.instructions = layerInstructions
            videoComposition = vc
        }
        return Result(
            composition: composition, videoComposition: videoComposition, audioMix: audioMix,
            renderSize: renderSize ?? CGSize(width: 1280, height: 720)
        )
    }
}

/// Reads the metadata the editor needs from a media file.
enum MediaProbe {
    static func asset(for url: URL, sourceJobId: String? = nil) async throws -> Asset {
        let media = AVURLAsset(url: url)
        let duration = try await media.load(.duration).seconds
        let videoTrack = try await media.loadTracks(withMediaType: .video).first
        let hasAudio = !(try await media.loadTracks(withMediaType: .audio)).isEmpty
        var width: Int?
        var height: Int?
        if let videoTrack {
            let size = try await videoTrack.load(.naturalSize).applying(try await videoTrack.load(.preferredTransform))
            width = Int(abs(size.width))
            height = Int(abs(size.height))
        }
        return Asset(
            path: url.path,
            kind: videoTrack == nil ? .audio : .video,
            name: url.deletingPathExtension().lastPathComponent,
            duration: duration.isFinite ? duration : 0,
            hasAudio: hasAudio,
            width: width,
            height: height,
            sourceJobId: sourceJobId
        )
    }
}
