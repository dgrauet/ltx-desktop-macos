import SwiftUI
import AVKit

struct HistoryView: View {
    @StateObject private var vm = HistoryViewModel()
    @State private var showDeleteConfirmation = false
    @State private var videoToDelete: VideoItem? = nil

    let columns = [
        GridItem(.adaptive(minimum: 180, maximum: 220), spacing: 12)
    ]

    var body: some View {
        HSplitView {
            gridPanel
                .frame(minWidth: 400)

            detailPanel
                .frame(minWidth: 320, maxWidth: .infinity)
        }
        .onAppear { vm.loadVideos() }
        .toolbar {
            ToolbarItem {
                Button(action: { vm.loadVideos() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh")
            }
        }
        .confirmationDialog(
            "Delete Video",
            isPresented: $showDeleteConfirmation,
            presenting: videoToDelete
        ) { item in
            Button("Delete", role: .destructive) {
                vm.deleteVideo(item)
            }
            Button("Cancel", role: .cancel) {}
        } message: { item in
            Text("Are you sure you want to delete \"\(item.displayName)\"? This cannot be undone.")
        }
    }

    // MARK: - Grid Panel

    private var gridPanel: some View {
        Group {
            if vm.videos.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(vm.videos) { item in
                            VideoCard(
                                item: item,
                                thumbnail: vm.thumbnails[item.jobId],
                                isSelected: vm.selectedVideo?.id == item.id
                            )
                            .onTapGesture {
                                vm.selectedVideo = item
                            }
                            .contextMenu {
                                Button(role: .destructive) {
                                    videoToDelete = item
                                    showDeleteConfirmation = true
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                    .padding(16)
                }
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "film.stack")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text("No videos yet.")
                .font(.title3)
                .fontWeight(.medium)
            Text("Generate your first video in the Generation tab.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Detail Panel

    private var detailPanel: some View {
        Group {
            if let item = vm.selectedVideo {
                VideoDetailView(
                    item: item,
                    onDelete: {
                        videoToDelete = item
                        showDeleteConfirmation = true
                    }
                )
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "arrow.left")
                        .font(.system(size: 32))
                        .foregroundStyle(.tertiary)
                    Text("Select a video to preview")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

// MARK: - Video Card

struct VideoCard: View {
    let item: VideoItem
    let thumbnail: NSImage?
    let isSelected: Bool

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Thumbnail
            ZStack {
                if let img = thumbnail {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 180, height: 110)
                        .clipped()
                } else {
                    Rectangle()
                        .fill(Color(.controlBackgroundColor))
                        .frame(width: 180, height: 110)
                    Image(systemName: "film")
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Metadata
            VStack(alignment: .leading, spacing: 2) {
                Text(item.jobId)
                    .font(.caption)
                    .fontWeight(.medium)
                    .lineLimit(1)

                HStack(spacing: 4) {
                    Text(item.resolutionLabel)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 4))

                    Spacer()

                    Text(Self.dateFormatter.string(from: item.createdAt))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, 4)
        }
        .padding(8)
        .background(
            isSelected
                ? Color.accentColor.opacity(0.15)
                : Color(.controlBackgroundColor).opacity(0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(
                    isSelected ? Color.accentColor : Color.clear,
                    lineWidth: 2
                )
        )
    }
}

// MARK: - Video Detail View

struct VideoDetailView: View {
    let item: VideoItem
    let onDelete: () -> Void

    @State private var player: AVPlayer?

    private static let fullDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .medium
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Player
            if let player = player {
                VideoPlayer(player: player)
                    .frame(height: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding([.top, .horizontal], 16)
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(.windowBackgroundColor).opacity(0.5))
                        .frame(height: 260)
                    ProgressView()
                }
                .padding([.top, .horizontal], 16)
            }

            // Metadata
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Divider()
                        .padding(.top, 12)

                    metadataRow(label: "Job ID", value: item.jobId)
                    metadataRow(label: "Resolution", value: item.resolutionLabel)
                    metadataRow(label: "Frames", value: "\(item.numFrames) @ \(item.fps) fps")
                    if item.durationSeconds > 0 {
                        metadataRow(label: "Duration", value: String(format: "%.1fs", item.durationSeconds))
                    }
                    metadataRow(label: "Seed", value: "\(item.seed)")
                    metadataRow(label: "Created", value: Self.fullDateFormatter.string(from: item.createdAt))

                    if item.prompt != "(no prompt)" {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Prompt")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(item.prompt)
                                .font(.body)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.top, 4)
                    }

                    Divider()
                        .padding(.top, 4)

                    Button(role: .destructive, action: onDelete) {
                        Label("Delete Video", systemImage: "trash")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .padding(.bottom, 8)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
        }
        .onAppear {
            player = AVPlayer(url: item.fileURL)
        }
        .onDisappear {
            player?.pause()
            player = nil
        }
        .onChange(of: item.id) { _, _ in
            player?.pause()
            player = AVPlayer(url: item.fileURL)
        }
    }

    private func metadataRow(label: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
