import SwiftUI

/// Dedicated Queue tab showing all queued and running generation jobs.
struct QueueView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var vm = QueueViewModel()

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Generation Queue")
                    .font(.title2)
                    .fontWeight(.semibold)

                if vm.runningCount > 0 || vm.queuedCount > 0 {
                    queueSummaryBadge
                }

                Spacer()

                Button(action: {
                    Task { await vm.explicitRefresh(service: backendService) }
                }) {
                    if vm.isLoading {
                        ProgressView()
                            .scaleEffect(0.7)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .buttonStyle(.borderless)
                .help("Refresh queue")
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            // Content
            if vm.entries.isEmpty {
                emptyState
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(vm.entries) { entry in
                            QueueJobRow(
                                entry: entry,
                                onCancel: {
                                    Task { await vm.cancelJob(jobId: entry.jobId, service: backendService) }
                                },
                                onPriorityChange: { newPriority in
                                    Task { await vm.changeJobPriority(jobId: entry.jobId, priority: newPriority, service: backendService) }
                                }
                            )
                            Divider()
                                .padding(.horizontal, 16)
                        }
                    }
                    .padding(.vertical, 8)
                }
            }

            // Error banner
            if let error = vm.errorMessage {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(error)
                        .font(.caption)
                    Spacer()
                    Button("Dismiss") {
                        vm.errorMessage = nil
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.orange.opacity(0.1))
            }
        }
        .onAppear {
            vm.startPolling(service: backendService)
        }
        .onDisappear {
            vm.stopPolling()
        }
    }

    // MARK: - Summary Badge

    private var queueSummaryBadge: some View {
        HStack(spacing: 4) {
            if vm.runningCount > 0 {
                Circle()
                    .fill(.green)
                    .frame(width: 6, height: 6)
                Text("\(vm.runningCount) running")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if vm.queuedCount > 0 {
                Circle()
                    .fill(.orange)
                    .frame(width: 6, height: 6)
                Text("\(vm.queuedCount) queued")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Color(.controlBackgroundColor).opacity(0.5))
        .clipShape(Capsule())
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "list.number")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text("Queue is empty")
                .font(.title3)
                .fontWeight(.medium)
            Text("Submit generations from the Generation tab.\nUse \"Add to Queue\" to batch multiple jobs.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Queue Job Row

struct QueueJobRow: View {
    let entry: QueueEntry
    let onCancel: () -> Void
    let onPriorityChange: (String) -> Void

    var body: some View {
        HStack(spacing: 12) {
            // State indicator
            stateIndicator

            // Job info
            VStack(alignment: .leading, spacing: 4) {
                // Type badge + prompt
                HStack(spacing: 6) {
                    Text(entry.jobType.uppercased())
                        .font(.caption2)
                        .fontWeight(.bold)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(jobTypeColor.opacity(0.15))
                        .foregroundStyle(jobTypeColor)
                        .clipShape(RoundedRectangle(cornerRadius: 4))

                    Text(entry.prompt)
                        .font(.subheadline)
                        .lineLimit(2)
                        .foregroundStyle(.primary)
                }

                // Status line: priority, position/progress, ETA
                HStack(spacing: 10) {
                    // Priority
                    HStack(spacing: 3) {
                        Image(systemName: priorityIcon)
                            .font(.caption2)
                        Text(entry.priority.capitalized)
                            .font(.caption)
                    }
                    .foregroundStyle(priorityColor)

                    // Position or progress
                    if entry.state == "running" {
                        if let progress = entry.progress {
                            HStack(spacing: 4) {
                                ProgressView(value: progress)
                                    .frame(width: 60)
                                Text("\(Int(progress * 100))%")
                                    .font(.caption)
                                    .monospacedDigit()
                            }
                            .foregroundStyle(.secondary)
                        } else {
                            HStack(spacing: 3) {
                                ProgressView()
                                    .scaleEffect(0.5)
                                Text("Running")
                                    .font(.caption)
                            }
                            .foregroundStyle(.secondary)
                        }
                    } else if entry.state == "queued" {
                        Text("Position \(entry.position)")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }

                    // ETA
                    if let eta = entry.etaSeconds {
                        Text("~\(formatETA(eta))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    // Status message
                    if let status = entry.status {
                        Text(status)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }

                    Spacer()
                }
            }

            Spacer()

            // Action buttons
            HStack(spacing: 4) {
                if entry.state == "queued" {
                    // Priority menu
                    Menu {
                        Button(action: { onPriorityChange("high") }) {
                            Label("High Priority", systemImage: "arrow.up.circle")
                        }
                        Button(action: { onPriorityChange("normal") }) {
                            Label("Normal Priority", systemImage: "minus.circle")
                        }
                        Button(action: { onPriorityChange("low") }) {
                            Label("Low Priority", systemImage: "arrow.down.circle")
                        }
                    } label: {
                        Image(systemName: "arrow.up.arrow.down")
                            .font(.system(size: 13))
                            .frame(width: 28, height: 28)
                            .contentShape(Rectangle())
                    }
                    .menuStyle(.borderlessButton)
                    .frame(width: 28)
                    .help("Change priority")
                }

                if entry.state == "queued" || entry.state == "running" {
                    Button(action: onCancel) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 16))
                            .foregroundStyle(.red.opacity(0.8))
                    }
                    .buttonStyle(.borderless)
                    .help("Cancel job")
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(entry.state == "running" ? Color.green.opacity(0.04) : Color.clear)
    }

    // MARK: - Helpers

    private var stateIndicator: some View {
        Group {
            switch entry.state {
            case "running":
                Circle()
                    .fill(.green)
                    .frame(width: 10, height: 10)
                    .overlay(
                        Circle()
                            .stroke(.green.opacity(0.3), lineWidth: 3)
                    )
            case "queued":
                Circle()
                    .fill(.orange)
                    .frame(width: 10, height: 10)
            case "completed":
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.blue)
            case "failed":
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.red)
            case "cancelled":
                Image(systemName: "minus.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.gray)
            default:
                Circle()
                    .fill(.secondary)
                    .frame(width: 10, height: 10)
            }
        }
    }

    private var jobTypeColor: Color {
        switch entry.jobType {
        case "t2v": return .blue
        case "i2v": return .green
        case "preview": return .orange
        case "retake": return .purple
        case "extend": return .cyan
        default: return .secondary
        }
    }

    private var priorityIcon: String {
        switch entry.priority {
        case "high": return "arrow.up.circle.fill"
        case "low": return "arrow.down.circle"
        default: return "minus.circle"
        }
    }

    private var priorityColor: Color {
        switch entry.priority {
        case "high": return .red
        case "low": return .secondary
        default: return .primary
        }
    }

    private func formatETA(_ seconds: Double) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else if seconds < 3600 {
            let m = Int(seconds / 60)
            let s = Int(seconds) % 60
            return "\(m)m \(s)s"
        } else {
            let h = Int(seconds / 3600)
            let m = Int(seconds.truncatingRemainder(dividingBy: 3600) / 60)
            return "\(h)h \(m)m"
        }
    }
}
