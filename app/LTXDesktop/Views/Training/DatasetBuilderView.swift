import SwiftUI
import UniformTypeIdentifiers

private let minClips = 5

// MARK: - DatasetBuilderView

struct DatasetBuilderView: View {
    @EnvironmentObject var backendService: BackendService
    @EnvironmentObject var vm: TrainingViewModel

    @State private var isDropTargeted = false
    @State private var showNewDatasetAlert = false
    @State private var newDatasetId = ""
    @State private var showCaptionGuidance = false
    @State private var isSaving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            headerBar
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    datasetSelector
                    dropZone
                    if !vm.manifestRows.isEmpty {
                        clipsTable
                    }
                    warningsArea
                    captionGuidance
                    saveButton
                }
                .padding(16)
            }
        }
        .task { await vm.loadDatasets(using: backendService) }
        .alert("New Dataset", isPresented: $showNewDatasetAlert) {
            TextField("Dataset ID (e.g. my-dataset)", text: $newDatasetId)
            Button("Create") {
                let id = newDatasetId.trimmingCharacters(in: .whitespaces)
                guard !id.isEmpty else { return }
                Task { await vm.createDataset(id, using: backendService) }
                newDatasetId = ""
            }
            Button("Cancel", role: .cancel) { newDatasetId = "" }
        } message: {
            Text("Enter a short identifier for the new dataset.")
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            Text("Dataset Builder")
                .font(.title2)
                .fontWeight(.semibold)
            Spacer()
            Button {
                Task { await vm.loadDatasets(using: backendService) }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .help("Refresh datasets")
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }

    // MARK: - Dataset Selector

    private var datasetSelector: some View {
        HStack(spacing: 10) {
            if vm.datasets.isEmpty {
                Text("No datasets — create one first.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            } else {
                Picker("Dataset", selection: Binding(
                    get: { vm.selectedDatasetId ?? "" },
                    set: { vm.selectedDatasetId = $0.isEmpty ? nil : $0 }
                )) {
                    Text("Select…").tag("")
                    ForEach(vm.datasets) { dataset in
                        Text(dataset.id).tag(dataset.id)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(maxWidth: 240)
            }

            Button {
                showNewDatasetAlert = true
            } label: {
                Label("New dataset", systemImage: "plus")
            }
            .buttonStyle(.bordered)
        }
    }

    // MARK: - Drop Zone

    private var dropZone: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    style: StrokeStyle(lineWidth: 2, dash: [6])
                )
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(isDropTargeted ? Color.accentColor.opacity(0.08) : Color.clear)
                )
                .frame(height: 100)

            VStack(spacing: 8) {
                Image(systemName: "film.stack")
                    .font(.system(size: 28))
                    .foregroundStyle(isDropTargeted ? Color.accentColor : Color.secondary)
                Text("Drop video clips here")
                    .font(.callout)
                    .foregroundStyle(isDropTargeted ? .primary : .secondary)
                Button("Add clips…") {
                    openClipPanel()
                }
                .buttonStyle(.borderless)
                .foregroundStyle(Color.accentColor)
                .font(.callout)
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers: providers)
        }
        .disabled(vm.selectedDatasetId == nil)
        .opacity(vm.selectedDatasetId == nil ? 0.4 : 1.0)
        .help(vm.selectedDatasetId == nil ? "Select a dataset first" : "Drop video files to add clips")
    }

    // MARK: - Clips Table

    private var clipsTable: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Clips (\(vm.manifestRows.count))")
                    .font(.headline)
                if vm.manifestRows.count < minClips {
                    badgeView("\(vm.manifestRows.count)/\(minClips)", color: .orange)
                } else {
                    badgeView("\(vm.manifestRows.count)", color: .green)
                }
            }

            VStack(spacing: 0) {
                // Header row
                HStack(spacing: 0) {
                    Text("Filename")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)
                        .frame(width: 200, alignment: .leading)
                        .padding(.horizontal, 8)
                    Divider().frame(height: 24)
                    Text("Caption")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                    Spacer()
                }
                .frame(height: 28)
                .background(Color.secondary.opacity(0.08))

                Divider()

                ForEach($vm.manifestRows) { $row in
                    HStack(spacing: 0) {
                        Text(row.video)
                            .font(.callout)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(width: 200, alignment: .leading)
                            .padding(.horizontal, 8)
                        Divider()
                        TextField("Add a caption…", text: $row.caption, axis: .vertical)
                            .textFieldStyle(.plain)
                            .font(.callout)
                            .lineLimit(2...4)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                    }
                    .frame(minHeight: 36)
                    Divider()
                }
            }
            .background(Color.secondary.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
            )
        }
    }

    // MARK: - Warnings Area

    @ViewBuilder
    private var warningsArea: some View {
        if vm.manifestRows.count < minClips {
            warningRow(
                "Fewer than \(minClips) clips — training results will likely be poor.",
                severity: .warning
            )
        }

        ForEach(vm.manifestWarnings, id: \.self) { warning in
            warningRow(warning, severity: .warning)
        }

        let emptyCaptions = vm.manifestRows.filter { $0.caption.trimmingCharacters(in: .whitespaces).isEmpty }.count
        if emptyCaptions > 0 {
            warningRow("\(emptyCaptions) clip(s) have no caption — add captions for better LoRA quality.", severity: .info)
        }
    }

    // MARK: - Caption Guidance

    private var captionGuidance: some View {
        DisclosureGroup("Caption guidance", isExpanded: $showCaptionGuidance) {
            VStack(alignment: .leading, spacing: 10) {
                Text("LTX-2.3 prompting structure — use one flowing paragraph ≤200 words:")
                    .font(.callout)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 6) {
                    guidanceRow("1", "Subject appearance", "Describe who or what is the main subject, their appearance, clothing, or notable features.")
                    guidanceRow("2", "Action / movement", "Describe exactly what the subject is doing, moving, or expressing.")
                    guidanceRow("3", "Environment / lighting", "Set the scene: location, time of day, lighting conditions, atmosphere.")
                    guidanceRow("4", "Camera angle / movement", "Specify framing (close-up, wide), movement (dolly, pan, static), and perspective.")
                    guidanceRow("5", "Style", "Visual style, color grading, film grain, aesthetic (cinematic, documentary, etc.).")
                    guidanceRow("6", "Audio elements", "Describe any ambient sound, music, or voice present in the scene.")
                }

                Text("Example:")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)

                Text("A middle-aged woman with curly red hair wearing a green wool coat walks through a misty forest path. She looks up and smiles as sunlight breaks through the canopy. Leaves fall gently around her. Camera slowly dollies forward. Cinematic, warm color grading, shallow depth of field. Soft orchestral music. Birds chirp.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(8)
                    .background(Color.secondary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .padding(.top, 8)
        }
        .font(.callout)
    }

    // MARK: - Save Button

    private var saveButton: some View {
        HStack {
            Spacer()
            Button {
                isSaving = true
                Task {
                    await vm.saveManifest(using: backendService)
                    isSaving = false
                }
            } label: {
                if isSaving {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Saving…")
                    }
                } else {
                    Label("Save manifest", systemImage: "square.and.arrow.down")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(vm.selectedDatasetId == nil || isSaving)
        }
    }

    // MARK: - Helpers

    private func openClipPanel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        panel.message = "Select video clips to add to the dataset"
        panel.prompt = "Add Clips"
        if panel.runModal() == .OK {
            let urls = panel.urls
            Task { await vm.addClips(urls, using: backendService) }
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        let lock = DispatchQueue(label: "datasetbuilder.drop")
        var urls: [URL] = []
        let group = DispatchGroup()

        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                group.enter()
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                    defer { group.leave() }
                    if let data = item as? Data,
                       let url = URL(dataRepresentation: data, relativeTo: nil) {
                        let ext = url.pathExtension.lowercased()
                        if ["mp4", "mov", "m4v", "avi", "mkv"].contains(ext) {
                            lock.sync { urls.append(url) }
                        }
                    }
                }
            }
        }

        group.notify(queue: .main) {
            guard !urls.isEmpty else { return }
            Task { await vm.addClips(urls, using: backendService) }
        }
        return true
    }

    private func badgeView(_ label: String, color: Color) -> some View {
        Text(label)
            .font(.caption2)
            .fontWeight(.bold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private func warningRow(_ message: String, severity: WarningSeverity) -> some View {
        HStack(spacing: 8) {
            Image(systemName: severity.iconName)
                .foregroundStyle(severity.color)
            Text(message)
                .font(.callout)
                .foregroundStyle(.primary)
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(severity.color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .strokeBorder(severity.color.opacity(0.25), lineWidth: 1)
        )
    }

    private func guidanceRow(_ number: String, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(number)
                .font(.caption2)
                .fontWeight(.bold)
                .frame(width: 16, height: 16)
                .background(Color.accentColor.opacity(0.15))
                .foregroundStyle(Color.accentColor)
                .clipShape(Circle())
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - WarningSeverity

private enum WarningSeverity {
    case warning, info

    var color: Color {
        switch self {
        case .warning: return .orange
        case .info: return .blue
        }
    }

    var iconName: String {
        switch self {
        case .warning: return "exclamationmark.triangle.fill"
        case .info: return "info.circle.fill"
        }
    }
}
