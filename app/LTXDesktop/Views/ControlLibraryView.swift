import SwiftUI
import AVKit

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

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let img = NSImage(contentsOfFile: item.thumbnailPath) {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fill)
                    .frame(height: 110).clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.2))
                    .frame(height: 110)
                    .overlay(Image(systemName: "square.stack.3d.up").foregroundStyle(.secondary))
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
                Button(role: .destructive) { onDelete() } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless).controlSize(.small)
            }
        }
        .padding(8)
        .background(Color(.controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
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
