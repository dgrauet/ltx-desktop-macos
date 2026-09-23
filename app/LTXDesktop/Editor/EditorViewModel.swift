import AVFoundation
import Combine
import Foundation

/// State for the Editor tab: the project list, the open project document
/// (with snapshot undo), playback of the edit, and clip regeneration.
@MainActor
final class EditorViewModel: ObservableObject {
    // Project list
    @Published private(set) var projects: [ProjectSummary] = []

    // Open document
    @Published private(set) var project: Project?
    @Published var selectedClipId: UUID?
    @Published var selectedAudioClipId: UUID?
    @Published private(set) var canUndo = false
    @Published private(set) var canRedo = false

    // Playback
    @Published var playhead: Double = 0
    @Published private(set) var isPlaying = false
    /// Timeline zoom, points per second.
    @Published var zoom: Double = 60
    let player = AVPlayer()

    // Status
    @Published var errorMessage: String?
    /// Non-nil while a long operation runs (regeneration, export).
    @Published private(set) var busyMessage: String?

    private let store: ProjectStore
    private var undoStack = UndoStack<Project>()
    private var saveTask: Task<Void, Never>?
    private var rebuildTask: Task<Void, Never>?
    private var timeObserver: Any?
    private var isScrubbing = false

    init(store: ProjectStore = ProjectStore()) {
        self.store = store
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(value: 1, timescale: 30), queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self, !self.isScrubbing else { return }
                self.playhead = time.seconds.isFinite ? time.seconds : 0
                self.isPlaying = self.player.rate != 0
            }
        }
        refreshProjects()
    }

    // MARK: - Projects

    func refreshProjects() {
        projects = store.list()
    }

    func createProject(name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        let project = Project(name: trimmed.isEmpty ? "Untitled Project" : trimmed)
        persistNow(project)
        open(project)
    }

    func openProject(_ id: UUID) {
        do {
            open(try store.load(id))
        } catch {
            errorMessage = "Could not open project: \(error.localizedDescription)"
        }
    }

    private func open(_ project: Project) {
        flushSave()
        self.project = project
        undoStack.clear()
        updateUndoState()
        selectedClipId = nil
        selectedAudioClipId = nil
        playhead = 0
        rebuildComposition(keepTime: false)
        refreshProjects()
    }

    func closeProject() {
        flushSave()
        player.pause()
        player.replaceCurrentItem(with: nil)
        project = nil
        refreshProjects()
    }

    func renameProject(_ name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        edit(undoable: false) { $0.name = trimmed }
    }

    func deleteProject(_ id: UUID) {
        if project?.id == id {
            saveTask?.cancel()
            project = nil
            player.replaceCurrentItem(with: nil)
        }
        do {
            try store.delete(id)
        } catch {
            errorMessage = "Could not delete project: \(error.localizedDescription)"
        }
        refreshProjects()
    }

    // MARK: - Editing

    /// Applies an edit to the open project. Failed edits leave it untouched and surface the error.
    func edit(undoable: Bool = true, _ body: (inout Project) throws -> Void) {
        guard var working = project else { return }
        let before = working
        do {
            try body(&working)
        } catch {
            errorMessage = error.localizedDescription
            return
        }
        guard working != before else { return }
        if undoable {
            undoStack.record(before)
        }
        working.modifiedAt = Date()
        project = working
        updateUndoState()
        documentChanged()
    }

    func undo() {
        guard let current = project, let previous = undoStack.undo(from: current) else { return }
        project = previous
        updateUndoState()
        documentChanged()
    }

    func redo() {
        guard let current = project, let next = undoStack.redo(from: current) else { return }
        project = next
        updateUndoState()
        documentChanged()
    }

    private func updateUndoState() {
        canUndo = undoStack.canUndo
        canRedo = undoStack.canRedo
    }

    private func documentChanged() {
        if let id = selectedClipId, project?.timeline.videoClips.contains(where: { $0.id == id }) != true {
            selectedClipId = nil
        }
        if let id = selectedAudioClipId, project?.timeline.audioClips.contains(where: { $0.id == id }) != true {
            selectedAudioClipId = nil
        }
        scheduleSave()
        rebuildComposition(keepTime: true)
    }

    var selectedClip: VideoClip? {
        guard let id = selectedClipId else { return nil }
        return project?.timeline.videoClips.first { $0.id == id }
    }

    var selectedAudioClip: AudioClip? {
        guard let id = selectedAudioClipId else { return nil }
        return project?.timeline.audioClips.first { $0.id == id }
    }

    // MARK: - Media

    /// Adds files to the asset bin (media is referenced in place, never copied).
    func importMedia(_ urls: [URL]) async {
        for url in urls {
            do {
                let asset = try await MediaProbe.asset(for: url)
                edit(undoable: false) { $0.addAsset(asset) }
            } catch {
                errorMessage = "Could not import \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    /// Adds a generated clip to the open project (creating one if needed) and appends it to the timeline.
    func sendGeneratedClip(path: String, jobId: String?, projectName: String = "Untitled Project") async {
        if project == nil {
            createProject(name: projectName)
        }
        do {
            let probed = try await MediaProbe.asset(for: URL(fileURLWithPath: path), sourceJobId: jobId)
            edit { project in
                let asset = project.addAsset(probed)
                try project.insertVideoClip(assetId: asset.id)
            }
        } catch {
            errorMessage = "Could not add the clip: \(error.localizedDescription)"
        }
    }

    /// Video assets go at the playhead on the video track; audio at the playhead on the music track.
    func addToTimeline(_ asset: Asset) {
        edit { project in
            if asset.kind == .video {
                try project.insertVideoClip(assetId: asset.id, atTime: playhead)
            } else {
                selectedAudioClipId = try project.addAudioClip(assetId: asset.id, start: playhead)
            }
        }
    }

    func removeAsset(_ id: UUID) {
        edit { try $0.removeAsset(id) }
    }

    // MARK: - Timeline commands

    func splitAtPlayhead() {
        edit { selectedClipId = try $0.splitVideoClip(atTime: playhead) }
    }

    func deleteSelection() {
        if let id = selectedAudioClipId {
            edit { try $0.deleteAudioClip(id) }
        } else if let id = selectedClipId {
            edit { try $0.deleteVideoClip(id) }
        }
    }

    func moveClip(_ id: UUID, to index: Int) {
        edit { try $0.moveVideoClip(id, to: index) }
    }

    func trimSelected(inPoint: Double, outPoint: Double) {
        guard let id = selectedClipId else { return }
        edit { try $0.trimVideoClip(id, inPoint: inPoint, outPoint: outPoint) }
    }

    func setSelectedVolume(_ volume: Double) {
        if let id = selectedAudioClipId {
            edit { try $0.setAudioClipVolume(id, volume) }
        } else if let id = selectedClipId {
            edit { try $0.setVideoClipVolume(id, volume) }
        }
    }

    func moveAudioClip(_ id: UUID, start: Double) {
        edit { try $0.moveAudioClip(id, start: start) }
    }

    func activateTake(_ takeId: UUID) {
        guard let clipId = selectedClipId else { return }
        edit { try $0.activateTake(takeId, inClip: clipId) }
    }

    func setVideoTrackMuted(_ muted: Bool) {
        edit { $0.timeline.videoTrackMuted = muted }
    }

    // MARK: - Playback

    func togglePlay() {
        if player.rate != 0 {
            player.pause()
        } else {
            if let duration = project?.timeline.duration, playhead >= duration - 0.05 {
                seek(to: 0)
            }
            player.play()
        }
        isPlaying = player.rate != 0
    }

    func seek(to seconds: Double) {
        let clamped = max(0, min(seconds, project?.timeline.duration ?? 0))
        playhead = clamped
        player.seek(to: CompositionBuilder.time(clamped), toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func beginScrub() { isScrubbing = true }
    func endScrub() { isScrubbing = false }

    func step(frames: Int) {
        let fps = Double(project?.fps ?? 24)
        seek(to: playhead + Double(frames) / fps)
    }

    private func rebuildComposition(keepTime: Bool) {
        rebuildTask?.cancel()
        guard let snapshot = project else { return }
        let resumeAt = keepTime ? playhead : 0
        let wasPlaying = player.rate != 0
        rebuildTask = Task { [weak self] in
            do {
                let result = try await CompositionBuilder.build(snapshot)
                guard !Task.isCancelled, let self else { return }
                let item = AVPlayerItem(asset: result.composition)
                item.videoComposition = result.videoComposition
                item.audioMix = result.audioMix
                self.player.replaceCurrentItem(with: item)
                self.seek(to: resumeAt)
                if wasPlaying { self.player.play() }
            } catch {
                guard !Task.isCancelled else { return }
                self?.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Persistence

    private func scheduleSave() {
        saveTask?.cancel()
        guard let snapshot = project else { return }
        saveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard !Task.isCancelled else { return }
            self?.persistNow(snapshot)
        }
    }

    private func flushSave() {
        saveTask?.cancel()
        if let project { persistNow(project) }
    }

    private func persistNow(_ project: Project) {
        do {
            try store.save(project)
        } catch {
            errorMessage = "Could not save project: \(error.localizedDescription)"
        }
    }

    // MARK: - Regeneration (takes)

    /// Regenerates a time range of the selected clip's active take and adds the result as a new take.
    /// `start`/`end` are seconds within the take's media.
    func retakeSelected(prompt: String, start: Double, end: Double, steps: Int, service: BackendService) async {
        guard let clip = selectedClip, let project, let media = project.asset(clip.activeTake.assetId) else { return }
        let request = RetakeRequest(
            sourceVideoPath: media.path, prompt: prompt, startTimeS: start, endTimeS: end,
            steps: steps, seed: -1, fps: project.fps
        )
        let takeNumber = clip.takes.count
        await runRegeneration(label: "Retaking clip…", service: service) {
            try await service.generateRetake(request: request).jobId
        } onResult: { [weak self] asset in
            self?.edit { project in
                let added = project.addAsset(asset)
                try project.addTake(toClip: clip.id, assetId: added.id, label: "Retake \(takeNumber)")
            }
        }
    }

    /// Extends the selected clip's active take; the extended media becomes a new take played in full.
    func extendSelected(prompt: String, pixelFrames: Int, forward: Bool, steps: Int, service: BackendService) async {
        guard let clip = selectedClip, let project, let media = project.asset(clip.activeTake.assetId) else { return }
        let request = ExtendRequest(
            sourceVideoPath: media.path, prompt: prompt, direction: forward ? "forward" : "backward",
            extensionFrames: pixelFrames, steps: steps, seed: -1, fps: project.fps
        )
        let takeNumber = clip.takes.count
        await runRegeneration(label: "Extending clip…", service: service) {
            try await service.generateExtend(request: request).jobId
        } onResult: { [weak self] asset in
            self?.edit { project in
                let added = project.addAsset(asset)
                try project.addTake(toClip: clip.id, assetId: added.id, label: "Extension \(takeNumber)")
                try project.trimVideoClip(clip.id, inPoint: 0, outPoint: added.duration)
            }
        }
    }

    private func runRegeneration(
        label: String,
        service: BackendService,
        submit: () async throws -> String,
        onResult: (Asset) -> Void
    ) async {
        guard busyMessage == nil else { return }
        busyMessage = label
        defer { busyMessage = nil }
        do {
            let jobId = try await submit()
            while true {
                try await Task.sleep(nanoseconds: 2_000_000_000)
                let status = try await service.getJobStatus(jobId: jobId)
                if status.status == "completed", let result = status.result {
                    let asset = try await MediaProbe.asset(for: URL(fileURLWithPath: result.outputPath), sourceJobId: jobId)
                    onResult(asset)
                    return
                }
                if status.status == "failed" || status.status == "cancelled" {
                    errorMessage = status.error ?? "Generation \(status.status)"
                    return
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Export

    func exportMovie(to url: URL, presetName: String = AVAssetExportPresetHighestQuality) async {
        guard let project else { return }
        busyMessage = "Exporting movie…"
        defer { busyMessage = nil }
        do {
            let result = try await CompositionBuilder.build(project)
            guard let session = AVAssetExportSession(asset: result.composition, presetName: presetName) else {
                errorMessage = "This export preset is not available."
                return
            }
            session.videoComposition = result.videoComposition
            session.audioMix = result.audioMix
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            let fileType: AVFileType = url.pathExtension.lowercased() == "mov" ? .mov : .mp4
            if #available(macOS 15, *) {
                try await session.export(to: url, as: fileType)
            } else {
                session.outputURL = url
                session.outputFileType = fileType
                await session.export()
                if session.status != .completed {
                    throw session.error ?? CocoaError(.fileWriteUnknown)
                }
            }
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    func exportFCPXML(to url: URL) {
        guard let project else { return }
        do {
            try FCPXMLWriter.document(for: project).write(to: url, atomically: true, encoding: .utf8)
        } catch {
            errorMessage = "FCPXML export failed: \(error.localizedDescription)"
        }
    }
}
