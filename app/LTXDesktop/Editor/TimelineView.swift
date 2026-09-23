import SwiftUI

/// Two-lane timeline: the magnetic video track and the music track, a ruler, and the playhead.
/// Click to seek, drag a video clip to reorder, drag a music clip to move it, drag its edges to trim.
struct TimelineView: View {
    @EnvironmentObject var editor: EditorViewModel

    private let rulerHeight: CGFloat = 22
    private let videoLaneHeight: CGFloat = 64
    private let audioLaneHeight: CGFloat = 40
    private let leading: CGFloat = 12

    @State private var dragClipId: UUID?
    @State private var dragOffset: CGFloat = 0

    var body: some View {
        VStack(spacing: 0) {
            timelineToolbar
            Divider()
            ScrollView(.horizontal) {
                ZStack(alignment: .topLeading) {
                    VStack(alignment: .leading, spacing: 6) {
                        ruler
                        videoLane
                        audioLane
                    }
                    playheadLine
                }
                .frame(width: contentWidth, alignment: .leading)
                .padding(.bottom, 8)
            }
        }
        .background(Color(.underPageBackgroundColor))
    }

    private var contentWidth: CGFloat {
        let duration = max(editor.project?.timeline.duration ?? 0, 10) + 5
        return leading + CGFloat(duration * editor.zoom)
    }

    private func x(_ seconds: Double) -> CGFloat { leading + CGFloat(seconds * editor.zoom) }
    private func seconds(_ x: CGFloat) -> Double { max(0, Double(x - leading) / editor.zoom) }

    // MARK: Toolbar

    private var timelineToolbar: some View {
        HStack(spacing: 12) {
            Button { editor.splitAtPlayhead() } label: { Label("Split", systemImage: "scissors") }
                .keyboardShortcut("b", modifiers: .command)
                .help("Split the clip under the playhead (⌘B)")
            Button { editor.deleteSelection() } label: { Label("Delete", systemImage: "trash") }
                .keyboardShortcut(.delete, modifiers: [])
                .disabled(editor.selectedClipId == nil && editor.selectedAudioClipId == nil)
            Spacer()
            Image(systemName: "minus.magnifyingglass").foregroundStyle(.secondary)
            Slider(value: $editor.zoom, in: 10...300).frame(width: 140)
            Image(systemName: "plus.magnifyingglass").foregroundStyle(.secondary)
        }
        .buttonStyle(.borderless)
        .labelStyle(.titleAndIcon)
        .font(.caption)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    // MARK: Ruler

    private var ruler: some View {
        let duration = max(editor.project?.timeline.duration ?? 0, 10) + 5
        let step = rulerStep
        return Canvas { context, size in
            var t = 0.0
            while t <= duration {
                let px = x(t)
                context.stroke(Path { p in
                    p.move(to: CGPoint(x: px, y: size.height - 6))
                    p.addLine(to: CGPoint(x: px, y: size.height))
                }, with: .color(.secondary), lineWidth: 1)
                context.draw(
                    Text(formatTimecode(t)).font(.system(size: 9).monospacedDigit()).foregroundColor(.secondary),
                    at: CGPoint(x: px + 2, y: 6), anchor: .leading
                )
                t += step
            }
        }
        .frame(height: rulerHeight)
        .contentShape(Rectangle())
        .gesture(scrubGesture)
    }

    private var rulerStep: Double {
        let minSpacing: Double = 70
        for step in [0.5, 1, 2, 5, 10, 30, 60] where step * editor.zoom >= minSpacing {
            return step
        }
        return 120
    }

    private var scrubGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                editor.beginScrub()
                editor.seek(to: seconds(value.location.x))
            }
            .onEnded { _ in editor.endScrub() }
    }

    // MARK: Video lane

    private var videoLane: some View {
        let project = editor.project
        let clips = project?.timeline.videoClips ?? []
        let starts = project?.timeline.videoClipStarts ?? []
        return ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.secondary.opacity(0.08))
                .frame(width: contentWidth, height: videoLaneHeight)
                .onTapGesture { location in
                    editor.selectedClipId = nil
                    editor.seek(to: seconds(location.x))
                }
            ForEach(Array(clips.enumerated()), id: \.element.id) { index, clip in
                let name = project?.asset(clip.activeTake.assetId)?.name ?? "Missing"
                let isDragging = dragClipId == clip.id
                clipBox(
                    title: name,
                    subtitle: clip.takes.count > 1 ? clip.activeTake.label : nil,
                    width: max(CGFloat(clip.duration * editor.zoom), 4),
                    height: videoLaneHeight,
                    color: .accentColor,
                    selected: editor.selectedClipId == clip.id
                )
                .overlay(alignment: .leading) { trimHandle(clip.id, leadingEdge: true) }
                .overlay(alignment: .trailing) { trimHandle(clip.id, leadingEdge: false) }
                .offset(x: x(starts[index]) + (isDragging ? dragOffset : 0))
                .zIndex(isDragging ? 1 : 0)
                .onTapGesture {
                    editor.selectedAudioClipId = nil
                    editor.selectedClipId = clip.id
                }
                .gesture(reorderGesture(clip: clip, index: index, starts: starts, clips: clips))
            }
        }
        .frame(height: videoLaneHeight)
    }

    private func reorderGesture(clip: VideoClip, index: Int, starts: [Double], clips: [VideoClip]) -> some Gesture {
        DragGesture(minimumDistance: 6)
            .onChanged { value in
                dragClipId = clip.id
                dragOffset = value.translation.width
            }
            .onEnded { value in
                defer {
                    dragClipId = nil
                    dragOffset = 0
                }
                let center = starts[index] + clip.duration / 2 + Double(value.translation.width) / editor.zoom
                var target = clips.count - 1
                for (i, s) in starts.enumerated() where center < s + clips[i].duration / 2 {
                    target = i
                    break
                }
                if target != index {
                    editor.moveClip(clip.id, to: target)
                }
                editor.selectedClipId = clip.id
            }
    }

    private func trimHandle(_ clipId: UUID, leadingEdge: Bool) -> some View {
        Rectangle()
            .fill(Color.white.opacity(0.001))
            .frame(width: 8)
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
            .gesture(
                DragGesture(minimumDistance: 2)
                    .onEnded { value in
                        guard let clip = editor.project?.timeline.videoClips.first(where: { $0.id == clipId }) else { return }
                        let delta = Double(value.translation.width) / editor.zoom
                        editor.selectedClipId = clipId
                        editor.selectedAudioClipId = nil
                        if leadingEdge {
                            editor.trimSelected(inPoint: clip.inPoint + delta, outPoint: clip.outPoint)
                        } else {
                            editor.trimSelected(inPoint: clip.inPoint, outPoint: clip.outPoint + delta)
                        }
                    }
            )
    }

    // MARK: Audio lane

    private var audioLane: some View {
        let project = editor.project
        let clips = project?.timeline.audioClips ?? []
        return ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.secondary.opacity(0.05))
                .frame(width: contentWidth, height: audioLaneHeight)
                .overlay(alignment: .leading) {
                    if clips.isEmpty {
                        Text("Music — add audio from the media bin")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.leading, leading + 4)
                    }
                }
                .onTapGesture { editor.selectedAudioClipId = nil }
            ForEach(clips) { clip in
                let isDragging = dragClipId == clip.id
                clipBox(
                    title: project?.asset(clip.assetId)?.name ?? "Missing",
                    subtitle: nil,
                    width: max(CGFloat(clip.duration * editor.zoom), 4),
                    height: audioLaneHeight,
                    color: .green,
                    selected: editor.selectedAudioClipId == clip.id
                )
                .offset(x: x(clip.start) + (isDragging ? dragOffset : 0))
                .onTapGesture {
                    editor.selectedClipId = nil
                    editor.selectedAudioClipId = clip.id
                }
                .gesture(
                    DragGesture(minimumDistance: 4)
                        .onChanged { value in
                            dragClipId = clip.id
                            dragOffset = value.translation.width
                        }
                        .onEnded { value in
                            dragClipId = nil
                            dragOffset = 0
                            editor.moveAudioClip(clip.id, start: clip.start + Double(value.translation.width) / editor.zoom)
                            editor.selectedAudioClipId = clip.id
                        }
                )
            }
        }
        .frame(height: audioLaneHeight)
    }

    // MARK: Pieces

    private func clipBox(title: String, subtitle: String?, width: CGFloat, height: CGFloat, color: Color, selected: Bool) -> some View {
        RoundedRectangle(cornerRadius: 5)
            .fill(color.opacity(0.35))
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .strokeBorder(selected ? Color.white : color.opacity(0.8), lineWidth: selected ? 2 : 1)
            )
            .overlay(alignment: .topLeading) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.caption2.weight(.medium)).lineLimit(1)
                    if let subtitle {
                        Text(subtitle).font(.system(size: 9)).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                .padding(4)
            }
            .frame(width: width, height: height)
            .clipped()
    }

    private var playheadLine: some View {
        Rectangle()
            .fill(Color.red)
            .frame(width: 1.5)
            .frame(height: rulerHeight + videoLaneHeight + audioLaneHeight + 12)
            .offset(x: x(editor.playhead))
            .allowsHitTesting(false)
    }
}
