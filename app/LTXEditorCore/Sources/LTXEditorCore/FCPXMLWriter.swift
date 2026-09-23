import Foundation

/// Exports a project's timeline as FCPXML 1.10 (Final Cut Pro, DaVinci Resolve).
///
/// Video clips form the primary storyline (spine); music clips are connected
/// clips (lane -1) attached to whichever spine item covers their start — a
/// trailing gap is added when music runs past the last video clip.
public enum FCPXMLWriter {

    public static func document(for project: Project) -> String {
        let fps = max(project.fps, 1)
        func frames(_ seconds: Double) -> Int { Int((seconds * Double(fps)).rounded()) }
        func time(_ seconds: Double) -> String {
            let f = frames(seconds)
            return f == 0 ? "0s" : "\(f)/\(fps)s"
        }

        // Resources: one format + one asset per referenced media file.
        let usedIds = Set(
            project.timeline.videoClips.map(\.activeTake.assetId) + project.timeline.audioClips.map(\.assetId)
        )
        let assets = project.assets.filter { usedIds.contains($0.id) }
        var resourceIds: [UUID: String] = [:]
        for (i, asset) in assets.enumerated() {
            resourceIds[asset.id] = "r\(i + 2)"
        }
        let firstVideo = assets.first { $0.kind == .video }
        let width = firstVideo?.width ?? 1920
        let height = firstVideo?.height ?? 1080

        var xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE fcpxml>
        <fcpxml version="1.10">
          <resources>
            <format id="r1" name="FFVideoFormatRateUndefined" frameDuration="1/\(fps)s" width="\(width)" height="\(height)"/>

        """
        for asset in assets {
            let hasVideo = asset.kind == .video ? "1" : "0"
            let hasAudio = asset.hasAudio || asset.kind == .audio ? "1" : "0"
            let format = asset.kind == .video ? " format=\"r1\"" : ""
            xml += """
                <asset id="\(resourceIds[asset.id]!)" name="\(escape(asset.name))" start="0s" duration="\(time(asset.duration))" hasVideo="\(hasVideo)" hasAudio="\(hasAudio)"\(format)>
                  <media-rep kind="original-media" src="\(escape(URL(fileURLWithPath: asset.path).absoluteString))"/>
                </asset>

            """
        }

        // Spine items with their timeline span, so music can be connected to them.
        struct SpineItem {
            var open: String
            var close: String
            var timelineStart: Double
            var duration: Double
            var sourceStart: Double
            var children: [String] = []
        }
        var spine: [SpineItem] = []
        let starts = project.timeline.videoClipStarts
        for (i, clip) in project.timeline.videoClips.enumerated() {
            let asset = project.asset(clip.activeTake.assetId)
            let gain = project.timeline.videoTrackMuted ? 0 : clip.volume
            let volume = gain < 1 ? "\n            <adjust-volume amount=\"\(decibels(gain))\"/>" : ""
            spine.append(SpineItem(
                open: "<asset-clip ref=\"\(resourceIds[clip.activeTake.assetId] ?? "r1")\" name=\"\(escape(asset?.name ?? "Clip"))\" offset=\"\(time(starts[i]))\" start=\"\(time(clip.inPoint))\" duration=\"\(time(clip.duration))\">\(volume)",
                close: "</asset-clip>",
                timelineStart: starts[i], duration: clip.duration, sourceStart: clip.inPoint
            ))
        }
        let videoEnd = project.timeline.videoClips.reduce(0) { $0 + $1.duration }
        let totalDuration = project.timeline.duration
        if totalDuration - videoEnd >= 1.0 / Double(fps) {
            spine.append(SpineItem(
                open: "<gap name=\"Gap\" offset=\"\(time(videoEnd))\" start=\"0s\" duration=\"\(time(totalDuration - videoEnd))\">",
                close: "</gap>", timelineStart: videoEnd, duration: totalDuration - videoEnd, sourceStart: 0
            ))
        }

        for clip in project.timeline.audioClips.sorted(by: { $0.start < $1.start }) {
            guard let index = spine.lastIndex(where: { $0.timelineStart <= clip.start + 1e-9 }) else { continue }
            let parent = spine[index]
            let offset = parent.sourceStart + (clip.start - parent.timelineStart)
            let asset = project.asset(clip.assetId)
            let volume = clip.volume < 1 ? "<adjust-volume amount=\"\(decibels(clip.volume))\"/>" : ""
            spine[index].children.append(
                "<asset-clip ref=\"\(resourceIds[clip.assetId] ?? "r1")\" lane=\"-1\" name=\"\(escape(asset?.name ?? "Audio"))\" offset=\"\(time(offset))\" start=\"\(time(clip.inPoint))\" duration=\"\(time(clip.duration))\">\(volume)</asset-clip>"
            )
        }

        xml += """
          </resources>
          <library>
            <event name="LTX Desktop">
              <project name="\(escape(project.name))">
                <sequence format="r1" duration="\(time(totalDuration))" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                  <spine>

        """
        for item in spine {
            xml += "          \(item.open)\n"
            for child in item.children {
                xml += "            \(child)\n"
            }
            xml += "          \(item.close)\n"
        }
        xml += """
                  </spine>
                </sequence>
              </project>
            </event>
          </library>
        </fcpxml>

        """
        return xml
    }

    static func decibels(_ gain: Double) -> String {
        guard gain > 0 else { return "-96dB" }
        return String(format: "%.1fdB", 20 * log10(gain))
    }

    static func escape(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}
