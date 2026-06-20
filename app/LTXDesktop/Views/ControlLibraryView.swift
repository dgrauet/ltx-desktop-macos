import SwiftUI
import AVFoundation

/// Reusable grid of saved control videos. `onUse` applies an item; if nil, the
/// row shows no Use button (view-only context).
struct ControlLibraryGrid: View {
    @ObservedObject var vm: ControlLibraryViewModel
    var onUse: ((ControlLibraryItem) -> Void)?

    private let columns = [GridItem(.adaptive(minimum: 180, maximum: 220), spacing: 12)]

    var body: some View {
        Group {
            if vm.items.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "square.stack.3d.up")
                        .font(.system(size: 48)).foregroundStyle(.secondary)
                    Text("No saved control videos yet")
                        .foregroundStyle(.secondary)
                    Text("Pose/Depth extractions are saved here automatically.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(vm.items) { item in
                            ControlLibraryCard(item: item, onUse: onUse) {
                                Task { await vm.delete(item) }
                            }
                        }
                    }
                    .padding(12)
                }
            }
        }
        .task { await vm.load() }
    }
}

struct ControlLibraryCard: View {
    let item: ControlLibraryItem
    var onUse: ((ControlLibraryItem) -> Void)?
    var onDelete: () -> Void

    @State private var showDeleteConfirm = false
    @State private var showPreview = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Tappable thumbnail → plays the control video in a sheet.
            Group {
                if let img = NSImage(contentsOfFile: item.thumbnailPath) {
                    Image(nsImage: img).resizable().aspectRatio(contentMode: .fill)
                        .frame(height: 110).clipped()
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                } else {
                    RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.2))
                        .frame(height: 110)
                        .overlay(Image(systemName: "square.stack.3d.up").foregroundStyle(.secondary))
                }
            }
            .overlay(
                Image(systemName: "play.circle.fill")
                    .font(.title)
                    .foregroundStyle(.white.opacity(0.85))
                    .shadow(radius: 2)
            )
            .contentShape(Rectangle())
            .onTapGesture { showPreview = true }
            .help("Play this control video")
            .sheet(isPresented: $showPreview) {
                ControlVideoPreview(item: item) { showPreview = false }
            }
            HStack(spacing: 6) {
                Text(item.typeLabel).font(.caption2).fontWeight(.bold)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Color.indigo.opacity(0.15)).foregroundStyle(.indigo)
                    .clipShape(RoundedRectangle(cornerRadius: 3))
                Text("\(item.width)×\(item.height) · \(item.frameCount)f")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Text(item.sourceName).font(.caption2).lineLimit(1).foregroundStyle(.secondary)
            HStack {
                if let onUse {
                    Button("Use") { onUse(item) }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                }
                Spacer()
                Button(role: .destructive) { showDeleteConfirm = true } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless).controlSize(.small)
                .confirmationDialog("Delete this control video?", isPresented: $showDeleteConfirm, titleVisibility: .visible) {
                    Button("Delete", role: .destructive) { onDelete() }
                    Button("Cancel", role: .cancel) {}
                }
            }
        }
        .padding(8)
        .background(Color(.controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// Plays a saved control video in a sheet using the AppKit player.
struct ControlVideoPreview: View {
    let item: ControlLibraryItem
    var onDone: () -> Void

    private var player: AVPlayer? {
        FileManager.default.fileExists(atPath: item.videoPath)
            ? AVPlayer(url: URL(fileURLWithPath: item.videoPath)) : nil
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("\(item.typeLabel) · \(item.sourceName)")
                    .font(.headline).lineLimit(1)
                Spacer()
                Button("Done") { onDone() }
            }
            .padding()
            if let player {
                PlayerView(player: player)
                    .onAppear { player.play() }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle).foregroundStyle(.secondary)
                    Text("This control video is no longer available.")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: 640, height: 520)
    }
}

/// Dedicated-tab wrapper: applies a selection to the shared GenerationViewModel
/// and routes back to Generation.
struct ControlLibraryView: View {
    @EnvironmentObject var vm: GenerationViewModel
    @StateObject private var libraryVM = ControlLibraryViewModel()
    var onUse: () -> Void   // switch tab to generation

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Control Videos").font(.title2).fontWeight(.semibold)
                Spacer()
            }
            .padding(.horizontal, 16).padding(.top, 12).padding(.bottom, 8)
            Divider()
            ControlLibraryGrid(vm: libraryVM) { item in
                vm.applyLibraryItem(item)
                onUse()
            }
        }
    }
}
