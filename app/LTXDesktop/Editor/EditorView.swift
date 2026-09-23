import AVKit
import SwiftUI
import UniformTypeIdentifiers

/// Editor tab: project list when nothing is open, otherwise the editing workspace.
struct EditorView: View {
    @EnvironmentObject var editor: EditorViewModel

    var body: some View {
        Group {
            if editor.project != nil {
                EditorWorkspace()
            } else {
                ProjectsListView()
            }
        }
        .alert("Editor", isPresented: Binding(
            get: { editor.errorMessage != nil },
            set: { if !$0 { editor.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { editor.errorMessage = nil }
        } message: {
            Text(editor.errorMessage ?? "")
        }
    }
}

// MARK: - Projects list

private struct ProjectsListView: View {
    @EnvironmentObject var editor: EditorViewModel
    @State private var newName = ""
    @State private var pendingDelete: ProjectSummary?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Projects")
                    .font(.title2.weight(.semibold))
                Spacer()
                TextField("New project name", text: $newName)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 220)
                    .onSubmit(create)
                Button("New Project", systemImage: "plus", action: create)
                    .buttonStyle(.borderedProminent)
            }

            if editor.projects.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "film.stack")
                        .font(.system(size: 42))
                        .foregroundStyle(.secondary)
                    Text("No projects yet")
                        .font(.headline)
                    Text("Create a project, then add clips from History or your files — or use “Send to Editor” after a generation.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 420)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(editor.projects) { summary in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(summary.name).font(.body.weight(.medium))
                            Text("\(summary.clipCount) clip\(summary.clipCount == 1 ? "" : "s") · \(formatTimecode(summary.duration)) · edited \(summary.modifiedAt.formatted(.relative(presentation: .named)))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Open") { editor.openProject(summary.id) }
                            .buttonStyle(.bordered)
                        Button(role: .destructive) {
                            pendingDelete = summary
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .help("Delete project (media files are kept)")
                    }
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { editor.openProject(summary.id) }
                }
                .listStyle(.inset(alternatesRowBackgrounds: true))
            }
        }
        .padding(20)
        .onAppear { editor.refreshProjects() }
        .confirmationDialog(
            "Delete “\(pendingDelete?.name ?? "")”?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { summary in
            Button("Delete Project", role: .destructive) { editor.deleteProject(summary.id) }
        } message: { _ in
            Text("The project is removed. Its media files stay on disk.")
        }
    }

    private func create() {
        editor.createProject(name: newName)
        newName = ""
    }
}

// MARK: - Workspace

private struct EditorWorkspace: View {
    @EnvironmentObject var editor: EditorViewModel
    @EnvironmentObject var backendService: BackendService
    @State private var showRetake = false
    @State private var showExtend = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            HSplitView {
                AssetBinView()
                    .frame(minWidth: 200, idealWidth: 240, maxWidth: 340)
                VStack(spacing: 0) {
                    ProgramMonitor(player: editor.player)
                        .background(Color.black)
                    transportBar
                }
                .frame(minWidth: 360)
                InspectorView(showRetake: $showRetake, showExtend: $showExtend)
                    .frame(minWidth: 220, idealWidth: 260, maxWidth: 340)
            }
            .frame(minHeight: 260)
            Divider()
            TimelineView()
                .frame(minHeight: 170, idealHeight: 200)
        }
        .overlay(alignment: .top) {
            if let message = editor.busyMessage {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(message).font(.callout)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.regularMaterial, in: Capsule())
                .padding(.top, 52)
            }
        }
        .sheet(isPresented: $showRetake) {
            RegenerateSheet(mode: .retake).environmentObject(editor).environmentObject(backendService)
        }
        .sheet(isPresented: $showExtend) {
            RegenerateSheet(mode: .extend).environmentObject(editor).environmentObject(backendService)
        }
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            Button {
                editor.closeProject()
            } label: {
                Label("Projects", systemImage: "chevron.left")
            }
            .buttonStyle(.borderless)

            Text(editor.project?.name ?? "")
                .font(.headline)
                .lineLimit(1)

            Spacer()

            Button { editor.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                .disabled(!editor.canUndo)
                .keyboardShortcut("z", modifiers: .command)
                .help("Undo")
            Button { editor.redo() } label: { Image(systemName: "arrow.uturn.forward") }
                .disabled(!editor.canRedo)
                .keyboardShortcut("z", modifiers: [.command, .shift])
                .help("Redo")

            Divider().frame(height: 18)

            Menu {
                Button("Movie (MP4)…") { exportMovie() }
                Button("Final Cut Pro / Resolve (FCPXML)…") { exportFCPXML() }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(editor.project?.timeline.videoClips.isEmpty ?? true || editor.busyMessage != nil)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    private var transportBar: some View {
        HStack(spacing: 12) {
            Button { editor.step(frames: -1) } label: { Image(systemName: "backward.frame") }
                .keyboardShortcut(.leftArrow, modifiers: [])
            Button { editor.togglePlay() } label: {
                Image(systemName: editor.isPlaying ? "pause.fill" : "play.fill")
            }
            .keyboardShortcut(.space, modifiers: [])
            Button { editor.step(frames: 1) } label: { Image(systemName: "forward.frame") }
                .keyboardShortcut(.rightArrow, modifiers: [])
            Text("\(formatTimecode(editor.playhead)) / \(formatTimecode(editor.project?.timeline.duration ?? 0))")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Spacer()
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func exportMovie() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie]
        panel.nameFieldStringValue = "\(editor.project?.name ?? "Edit").mp4"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await editor.exportMovie(to: url) }
    }

    private func exportFCPXML() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "fcpxml") ?? .xml]
        panel.nameFieldStringValue = "\(editor.project?.name ?? "Edit").fcpxml"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        editor.exportFCPXML(to: url)
    }
}

// MARK: - Asset bin

private struct AssetBinView: View {
    @EnvironmentObject var editor: EditorViewModel
    @EnvironmentObject var backendService: BackendService
    @State private var history: [HistoryEntry] = []
    @State private var showHistory = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Media").font(.headline)
                Spacer()
                Menu {
                    Button("From History…") { showHistory = true }
                    Button("Import Files…") { importFiles() }
                } label: {
                    Image(systemName: "plus")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
            List {
                ForEach(editor.project?.assets ?? []) { asset in
                    HStack(spacing: 8) {
                        Image(systemName: asset.kind == .video ? "film" : "waveform")
                            .foregroundStyle(FileManager.default.fileExists(atPath: asset.path) ? Color.accentColor : .red)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(asset.name).lineLimit(1)
                            Text(formatTimecode(asset.duration))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button {
                            editor.addToTimeline(asset)
                        } label: {
                            Image(systemName: "plus.rectangle.on.rectangle")
                        }
                        .buttonStyle(.borderless)
                        .help(asset.kind == .video ? "Insert at playhead" : "Add to music track at playhead")
                    }
                    .contextMenu {
                        Button("Add to Timeline") { editor.addToTimeline(asset) }
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: asset.path)])
                        }
                        Divider()
                        Button("Remove from Project", role: .destructive) { editor.removeAsset(asset.id) }
                    }
                }
            }
            .listStyle(.sidebar)
        }
        .padding(10)
        .sheet(isPresented: $showHistory) {
            historyPicker
        }
    }

    private var historyPicker: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add from History").font(.headline)
            List(history, id: \.jobId) { entry in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(entry.prompt).lineLimit(2)
                        Text("\(entry.width)×\(entry.height) · \(entry.numFrames) frames · \(entry.generationType)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Add") {
                        Task { await editor.importMedia([URL(fileURLWithPath: entry.outputPath)]) }
                    }
                    .disabled(!FileManager.default.fileExists(atPath: entry.outputPath))
                }
            }
            .frame(minWidth: 520, minHeight: 360)
            HStack {
                Spacer()
                Button("Done") { showHistory = false }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .task {
            history = (try? await backendService.fetchHistory()) ?? []
        }
    }

    private func importFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.movie, .audio]
        guard panel.runModal() == .OK else { return }
        Task { await editor.importMedia(panel.urls) }
    }
}

// MARK: - Inspector

private struct InspectorView: View {
    @EnvironmentObject var editor: EditorViewModel
    @Binding var showRetake: Bool
    @Binding var showExtend: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let clip = editor.selectedClip, let project = editor.project {
                    videoClipInspector(clip, project: project)
                } else if let clip = editor.selectedAudioClip {
                    audioClipInspector(clip)
                } else {
                    Text("Select a clip on the timeline to edit it.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                Divider()
                Toggle("Mute clip audio", isOn: Binding(
                    get: { editor.project?.timeline.videoTrackMuted ?? false },
                    set: { editor.setVideoTrackMuted($0) }
                ))
                .toggleStyle(.switch)
                .controlSize(.small)
            }
            .padding(12)
        }
    }

    @ViewBuilder
    private func videoClipInspector(_ clip: VideoClip, project: Project) -> some View {
        let media = project.asset(clip.activeTake.assetId)
        Text("Clip").font(.headline)
        Text(media?.name ?? "Missing media")
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(2)

        LabeledContent("In") {
            TextField("", value: Binding(
                get: { clip.inPoint },
                set: { editor.trimSelected(inPoint: $0, outPoint: clip.outPoint) }
            ), format: .number.precision(.fractionLength(2)))
            .frame(width: 70)
        }
        LabeledContent("Out") {
            TextField("", value: Binding(
                get: { clip.outPoint },
                set: { editor.trimSelected(inPoint: clip.inPoint, outPoint: $0) }
            ), format: .number.precision(.fractionLength(2)))
            .frame(width: 70)
        }
        LabeledContent("Duration", value: formatTimecode(clip.duration))

        VStack(alignment: .leading, spacing: 4) {
            Text("Volume").font(.subheadline)
            Slider(value: Binding(
                get: { clip.volume },
                set: { editor.setSelectedVolume($0) }
            ), in: 0...1)
        }

        VStack(alignment: .leading, spacing: 6) {
            Text("Takes").font(.subheadline)
            Picker("Take", selection: Binding(
                get: { clip.activeTakeId },
                set: { editor.activateTake($0) }
            )) {
                ForEach(clip.takes) { take in
                    Text(take.label).tag(take.id)
                }
            }
            .labelsHidden()
            HStack {
                Button("Retake…") { showRetake = true }
                Button("Extend…") { showExtend = true }
            }
            .disabled(editor.busyMessage != nil)
            Text("Regenerated media is added as a new take; switch back any time.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func audioClipInspector(_ clip: AudioClip) -> some View {
        Text("Music clip").font(.headline)
        LabeledContent("Start") {
            TextField("", value: Binding(
                get: { clip.start },
                set: { editor.moveAudioClip(clip.id, start: $0) }
            ), format: .number.precision(.fractionLength(2)))
            .frame(width: 70)
        }
        LabeledContent("Duration", value: formatTimecode(clip.duration))
        VStack(alignment: .leading, spacing: 4) {
            Text("Volume").font(.subheadline)
            Slider(value: Binding(
                get: { clip.volume },
                set: { editor.setSelectedVolume($0) }
            ), in: 0...1)
        }
    }
}

// MARK: - Retake / Extend sheet

private struct RegenerateSheet: View {
    enum Mode { case retake, extend }

    let mode: Mode
    @EnvironmentObject var editor: EditorViewModel
    @EnvironmentObject var backendService: BackendService
    @Environment(\.dismiss) private var dismiss

    @State private var prompt = ""
    @State private var start: Double = 0
    @State private var end: Double = 1
    @State private var frames = 49
    @State private var forward = true
    @State private var steps = 8

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(mode == .retake ? "Retake segment" : "Extend clip").font(.headline)
            TextField("Describe the new content", text: $prompt, axis: .vertical)
                .lineLimit(3...6)
                .textFieldStyle(.roundedBorder)
            if mode == .retake {
                HStack {
                    Text("From")
                    TextField("", value: $start, format: .number.precision(.fractionLength(2))).frame(width: 70)
                    Text("to")
                    TextField("", value: $end, format: .number.precision(.fractionLength(2))).frame(width: 70)
                    Text("s (within the clip's media)").foregroundStyle(.secondary)
                }
            } else {
                Picker("Direction", selection: $forward) {
                    Text("After").tag(true)
                    Text("Before").tag(false)
                }
                .pickerStyle(.segmented)
                Picker("Frames", selection: $frames) {
                    ForEach([25, 49, 97], id: \.self) { n in Text("\(n)").tag(n) }
                }
            }
            Stepper("Steps: \(steps)", value: $steps, in: 1...50)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button(mode == .retake ? "Retake" : "Extend") {
                    let (p, s, e, f, fw, st) = (prompt, start, end, frames, forward, steps)
                    dismiss()
                    Task {
                        if mode == .retake {
                            await editor.retakeSelected(prompt: p, start: s, end: e, steps: st, service: backendService)
                        } else {
                            await editor.extendSelected(prompt: p, pixelFrames: f, forward: fw, steps: st, service: backendService)
                        }
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || (mode == .retake && end <= start))
            }
        }
        .padding(16)
        .frame(width: 420)
        .onAppear {
            if let clip = editor.selectedClip {
                start = clip.inPoint
                end = clip.outPoint
            }
        }
    }
}

/// Program monitor without AVKit transport controls (the editor has its own).
/// AVPlayerView rather than SwiftUI `VideoPlayer`, which has crashed on macOS 26 (see PlayerView).
private struct ProgramMonitor: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .none
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player {
            view.player = player
        }
    }
}

func formatTimecode(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds > 0 else { return "0:00.00" }
    let minutes = Int(seconds) / 60
    let rest = seconds - Double(minutes * 60)
    return String(format: "%d:%05.2f", minutes, rest)
}
