import SwiftUI
import AppKit

struct ExportSheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var backendService: BackendService
    let videoPath: String
    let clipName: String

    @State private var selectedFormat = "mp4"
    @State private var selectedCodec = "h264"
    @State private var selectedBitrate = "8M"
    @State private var isExporting = false
    @State private var exportedPath: String? = nil
    @State private var fcpxmlPath: String? = nil
    @State private var errorMessage: String? = nil

    private let formats = ["mp4", "mov"]
    private let bitrates = ["4M", "8M", "16M", "32M"]

    private var availableCodecs: [String] {
        if selectedFormat == "mp4" {
            return ["h264", "h265"]
        } else {
            return ["h264", "h265", "prores"]
        }
    }

    private func codecLabel(_ codec: String) -> String {
        switch codec {
        case "h264": return "H.264"
        case "h265": return "H.265/HEVC"
        case "prores": return "ProRes"
        default: return codec
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Title bar
            HStack {
                Text("Export Video")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
            }
            .padding([.top, .horizontal], 20)
            .padding(.bottom, 16)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Format & Codec section
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Format & Codec")
                            .font(.headline)

                        VStack(spacing: 0) {
                            // Format picker
                            HStack {
                                Text("Format")
                                    .font(.subheadline)
                                Spacer()
                                Picker("", selection: $selectedFormat) {
                                    Text("MP4").tag("mp4")
                                    Text("MOV").tag("mov")
                                }
                                .frame(width: 100)
                                .onChange(of: selectedFormat) { _, newFormat in
                                    // Reset codec if ProRes selected but format switched to MP4
                                    if newFormat == "mp4" && selectedCodec == "prores" {
                                        selectedCodec = "h264"
                                    }
                                }
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)

                            Divider().padding(.leading, 14)

                            // Codec picker
                            HStack {
                                Text("Codec")
                                    .font(.subheadline)
                                Spacer()
                                Picker("", selection: $selectedCodec) {
                                    ForEach(availableCodecs, id: \.self) { codec in
                                        Text(codecLabel(codec)).tag(codec)
                                    }
                                }
                                .frame(width: 140)
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)

                            Divider().padding(.leading, 14)

                            // Bitrate picker
                            HStack {
                                Text("Bitrate")
                                    .font(.subheadline)
                                Spacer()
                                Picker("", selection: $selectedBitrate) {
                                    ForEach(bitrates, id: \.self) { br in
                                        Text(br).tag(br)
                                    }
                                }
                                .frame(width: 100)
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                        }
                        .background(Color(.controlBackgroundColor).opacity(0.5))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }

                    // FCP / NLE Export section
                    VStack(alignment: .leading, spacing: 12) {
                        Text("FCP / NLE Export")
                            .font(.headline)

                        VStack(alignment: .leading, spacing: 10) {
                            Button(action: exportFCPXML) {
                                HStack(spacing: 6) {
                                    Image(systemName: "film.stack")
                                    Text("Export as FCPXML")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .disabled(isExporting)

                            if let path = fcpxmlPath {
                                HStack(spacing: 6) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.green)
                                    Text(path)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    Button("Reveal") {
                                        NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                                    }
                                    .buttonStyle(.plain)
                                    .font(.caption)
                                    .foregroundStyle(Color.accentColor)
                                }
                                .padding(8)
                                .background(Color.green.opacity(0.1))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                        .padding(14)
                        .background(Color(.controlBackgroundColor).opacity(0.5))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }

                    // Success state
                    if let path = exportedPath {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 6) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                                Text("Exported successfully")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                            }
                            Text(path)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .truncationMode(.middle)
                            Button("Reveal in Finder") {
                                NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    // Error state
                    if let error = errorMessage {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    // Progress indicator
                    if isExporting {
                        HStack(spacing: 8) {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Exporting...")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                    }
                }
                .padding(20)
            }

            Divider()

            // Action buttons
            HStack(spacing: 12) {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.escape)

                Spacer()

                Button(action: exportVideo) {
                    HStack(spacing: 6) {
                        if isExporting {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        Image(systemName: "square.and.arrow.up")
                        Text(isExporting ? "Exporting..." : "Export")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(16)
        }
        .frame(width: 480, height: 560)
    }

    // MARK: - Export Actions

    private func exportVideo() {
        isExporting = true
        exportedPath = nil
        errorMessage = nil

        let request = ExportVideoRequest(
            videoPath: videoPath,
            codec: selectedCodec,
            outputFormat: selectedFormat,
            bitrate: selectedBitrate
        )

        Task {
            do {
                let response = try await backendService.exportVideo(request: request)
                exportedPath = response.outputPath
            } catch {
                errorMessage = "Export failed: \(error.localizedDescription)"
            }
            isExporting = false
        }
    }

    private func exportFCPXML() {
        isExporting = true
        fcpxmlPath = nil
        errorMessage = nil

        let request = ExportFCPXMLRequest(
            videoPath: videoPath,
            clipName: clipName,
            frameRate: "24"
        )

        Task {
            do {
                let response = try await backendService.exportFCPXML(request: request)
                fcpxmlPath = response.outputPath
            } catch {
                errorMessage = "FCPXML export failed: \(error.localizedDescription)"
            }
            isExporting = false
        }
    }
}
