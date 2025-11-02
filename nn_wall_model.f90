module nn_wall_model
  use field, only : field_t
  use num_types, only : rp       ! double precision kind
  use json_module, only : json_file
  use dofmap, only : dofmap_t
  use coefs, only : coef_t
  use neko_config, only : NEKO_BCKND_DEVICE
  use wall_model, only : wall_model_t
  use field_registry, only : neko_field_registry
  use json_utils, only: json_get, json_get_or_default
  use utils, only : neko_error
  use iso_c_binding, only: c_int, c_float
  use, intrinsic :: iso_fortran_env, only: int32
  use torchfort
  implicit none
  private

  type, public, extends(wall_model_t) :: nn_wall_model_t
    character(len=:), allocatable :: model_name
    real(kind=c_float), allocatable :: scaler_X_mean(:)
    real(kind=c_float), allocatable :: scaler_X_scale(:)
    real(kind=c_float), allocatable :: scaler_Y_mean(:)
    real(kind=c_float), allocatable :: scaler_Y_scale(:)
    character(len=:), allocatable :: model_path
    character(len=:), allocatable :: scaler_X_mean_path
    character(len=:), allocatable :: scaler_X_scale_path
    character(len=:), allocatable :: scaler_Y_mean_path
    character(len=:), allocatable :: scaler_Y_scale_path
  contains
    procedure, pass :: init                => nn_wall_model_init
    procedure, pass :: init_from_components=> nn_wall_model_init_from_components
    procedure, pass :: free                => nn_wall_model_free
    procedure, pass :: compute             => nn_wall_model_compute
    procedure, private :: load_scalers
    procedure, private :: print_type_info
  end type nn_wall_model_t

contains

  subroutine print_type_info(this)
    class(nn_wall_model_t), intent(in) :: this
    write(*,*) "=== Type Information ==="
    write(*,*) "rp kind (double): ", kind(0.0_rp), " size (bytes):", storage_size(0.0_rp)/8
    write(*,*) "c_int kind: ", kind(c_int), " size (bytes):", storage_size(c_int)/8
    write(*,*) "c_float kind (single): ", kind(0.0_c_float), " size (bytes):", storage_size(0.0_c_float)/8
    write(*,*) "========================"
  end subroutine print_type_info


  subroutine nn_wall_model_init(this, coef, msk, facet, nu, h_index, json)
    class(nn_wall_model_t), intent(inout) :: this
    type(coef_t), intent(in) :: coef
    integer, intent(in) :: msk(:)
    integer, intent(in) :: facet(:)
    real(kind=rp), intent(in) :: nu
    integer, intent(in) :: h_index
    type(json_file), intent(inout) :: json

    character(len=:), allocatable :: mpath, smx, ssx, smy, ssy

    call json_get(json, "model_path", mpath)
    call json_get(json, "scaler_X_mean_path", smx)
    call json_get(json, "scaler_X_scale_path", ssx)
    call json_get(json, "scaler_Y_mean_path", smy)
    call json_get(json, "scaler_Y_scale_path", ssy)

    write(*,*) "nn_wall_model_init:"
    write(*,*) "  model_path: ", trim(mpath)
    write(*,*) "  scaler_X_mean_path: ", trim(smx)
    write(*,*) "  scaler_X_scale_path: ", trim(ssx)
    write(*,*) "  scaler_Y_mean_path: ", trim(smy)
    write(*,*) "  scaler_Y_scale_path: ", trim(ssy)

    call this%init_from_components(coef, msk, facet, nu, h_index, mpath, smx, ssx, smy, ssy)
  end subroutine nn_wall_model_init


  subroutine nn_wall_model_init_from_components(this, coef, msk, facet, nu, h_index, & 
                                               mpath, smx, ssx, smy, ssy)
    class(nn_wall_model_t), intent(inout) :: this
    type(coef_t), intent(in) :: coef
    integer, intent(in) :: msk(:)
    integer, intent(in) :: facet(:)
    real(kind=rp), intent(in) :: nu
    integer, intent(in) :: h_index
    character(len=*), intent(in) :: mpath, smx, ssx, smy, ssy

    integer :: ierr
    character(len=12) :: ierr_str
    character(len=:), allocatable :: model_config_path

    if (NEKO_BCKND_DEVICE == 1) then
      call neko_error("NN wall model only supported on CPU backend")
    end if

    call this%init_base(coef, msk, facet, nu, h_index)

    this%model_path = mpath
    this%scaler_X_mean_path = smx
    this%scaler_X_scale_path = ssx
    this%scaler_Y_mean_path = smy
    this%scaler_Y_scale_path = ssy
    this%nu = nu

    ! Assign a model name identifier used by TorchFort
    this%model_name = "model_scripted"

    ! === Added: create TorchFort model from YAML config before loading weights ===
    model_config_path = "/home/onkar/neko/src/wall_models/model_config.yaml"
    write(*,*) "Creating TorchFort model from config: ", trim(model_config_path)
    ierr = torchfort_create_model(this%model_name, trim(model_config_path), 0)  ! Device 0 (CPU)
    if (ierr /= 0) then
      write(ierr_str, '(I0)') ierr
      call neko_error("Failed to create TorchFort model, ierr=" // trim(ierr_str))
    end if

    write(*,*) "Loading TorchFort model weights from: ", trim(this%model_path)
    ierr = torchfort_load_model(this%model_name, trim(this%model_path))
    if (ierr /= 0) then
      write(ierr_str, '(I0)') ierr
      call neko_error("Failed to load TorchFort model, ierr=" // trim(ierr_str))
    end if
    write(*,*) "Model loaded successfully with name: ", trim(this%model_name)
    ! ===============================================================

    call this%load_scalers()
  end subroutine nn_wall_model_init_from_components


  subroutine load_scalers(this)
    class(nn_wall_model_t), intent(inout) :: this
    integer :: n_in, n_out

    n_in = 17
    n_out = 1

    allocate(this%scaler_X_mean(n_in))
    allocate(this%scaler_X_scale(n_in))
    allocate(this%scaler_Y_mean(n_out))
    allocate(this%scaler_Y_scale(n_out))

    write(*,*) "Loading scaler_X_mean from: ", trim(this%scaler_X_mean_path)
    call read_npy(this%scaler_X_mean_path, this%scaler_X_mean, n_in)

    write(*,*) "Loading scaler_X_scale from: ", trim(this%scaler_X_scale_path)
    call read_npy(this%scaler_X_scale_path, this%scaler_X_scale, n_in)

    write(*,*) "Loading scaler_Y_mean from: ", trim(this%scaler_Y_mean_path)
    call read_npy(this%scaler_Y_mean_path, this%scaler_Y_mean, n_out)

    write(*,*) "Loading scaler_Y_scale from: ", trim(this%scaler_Y_scale_path)
    call read_npy(this%scaler_Y_scale_path, this%scaler_Y_scale, n_out)
  end subroutine load_scalers


  subroutine nn_wall_model_free(this)
    class(nn_wall_model_t), intent(inout) :: this

    call this%free_base()
    ! TorchFort does not have explicit destroy in your provided snippets - omit or add if available
    if (allocated(this%scaler_X_mean)) deallocate(this%scaler_X_mean)
    if (allocated(this%scaler_X_scale)) deallocate(this%scaler_X_scale)
    if (allocated(this%scaler_Y_mean)) deallocate(this%scaler_Y_mean)
    if (allocated(this%scaler_Y_scale)) deallocate(this%scaler_Y_scale)
  end subroutine nn_wall_model_free


  subroutine nn_wall_model_compute(this, t, tstep)
    implicit none
    class(nn_wall_model_t), intent(inout) :: this
    real(kind=rp), intent(in) :: t
    integer, intent(in) :: tstep

    integer :: i, j, ierr
    real(kind=c_float) :: x_in_1d(17)
    real(kind=c_float) :: y_out_unscaled(1)
    real(kind=c_float), dimension(17,1) :: input_2d   ! features x batch=1
    real(kind=c_float), dimension(1,1)  :: output_2d  ! output x batch=1
    real(kind=c_float) :: u_sum, v_sum, w_sum, p_sum
    real(kind=c_float) :: n_nodes_real
    integer :: current_h_index
    integer :: k, base_idx
    real(kind=rp) :: ui, vi, wi, magu


    type(field_t), pointer :: u, v, w, p

    call this%print_type_info()

    u => neko_field_registry%get_field("u")
    v => neko_field_registry%get_field("v")
    w => neko_field_registry%get_field("w")
    p => neko_field_registry%get_field("p")

    write(*,*) "Starting nn_wall_model_compute for ", this%n_nodes, " nodes."

    do k = 0, 3
      current_h_index = this%h_index + k

      ! Initialize sums
      u_sum = 0.0_c_float
      v_sum = 0.0_c_float
      w_sum = 0.0_c_float
      p_sum = 0.0_c_float

      do i = 1, this%n_nodes
        ! Collect input features for node i
        u_sum = u_sum + real(u%x(this%ind_r(i), this%ind_s(i), current_h_index, this%ind_e(i)), kind=c_float)
        v_sum = v_sum + real(v%x(this%ind_r(i), this%ind_s(i), current_h_index, this%ind_e(i)), kind=c_float)
        w_sum = w_sum + real(w%x(this%ind_r(i), this%ind_s(i), current_h_index, this%ind_e(i)), kind=c_float)
        p_sum = p_sum + real(p%x(this%ind_r(i), this%ind_s(i), current_h_index, this%ind_e(i)), kind=c_float)
      end do

      base_idx = k*4
      n_nodes_real = real(this%n_nodes, kind=c_float)

      ! Compute averages
      n_nodes_real = real(this%n_nodes, kind=c_float)
      x_in_1d(base_idx +1) = u_sum / n_nodes_real
      x_in_1d(base_idx +2) = v_sum / n_nodes_real
      x_in_1d(base_idx +3) = w_sum / n_nodes_real
      x_in_1d(base_idx +4) = p_sum / n_nodes_real
      x_in_1d(17) = this%nu

    end do

    write(*,*) "Averaged features (before normalization):"
    write(*,'(17(f12.6,1x))') x_in_1d

    ! Normalize inputs
    do j = 1, 17
      x_in_1d(j) = (x_in_1d(j) - this%scaler_X_mean(j)) / this%scaler_X_scale(j)
    end do

    write(*,*) "Normalized averaged features:"
    write(*,'(17(f12.6,1x))') x_in_1d

    ! Prepare input tensor as 2D array (features, batch=1)
    input_2d(:,1) = x_in_1d
    output_2d = 0.0_c_float

    ! Call TorchFort inference function
    ierr = torchfort_inference(this%model_name, input_2d, output_2d)
    write(*,*) "TorchFort inference returned code: ", ierr

    ! De-normalize prediction
    y_out_unscaled(1) = output_2d(1,1) * this%scaler_Y_scale(1) + this%scaler_Y_mean(1)

    write(*,*) "Predicted y (unscaled, wall-averaged): ", y_out_unscaled(1)

    ! Assign the averaged prediction to all wall nodes
    do i = 1, this%n_nodes
      ! Sample the velocity at the wall node
      ui = u%x(this%ind_r(i), this%ind_s(i), this%h_index, this%ind_e(i))
      vi = v%x(this%ind_r(i), this%ind_s(i), this%h_index, this%ind_e(i))
      wi = w%x(this%ind_r(i), this%ind_s(i), this%h_index, this%ind_e(i))
      magu = sqrt(ui**2 + vi**2 + wi**2)
      if (magu > 0.0_c_float) then
        this%tau_x(i) = -(abs(y_out_unscaled(1))) * ui / magu
        this%tau_y(i) = 0.0_c_float
        this%tau_z(i) = 0.0_c_float
      else
        this%tau_x(i) = 0.0_c_float
        this%tau_y(i) = 0.0_c_float
        this%tau_z(i) = 0.0_c_float
      end if
    end do


    write(*,*) "Completed nn_wall_model_compute."
  end subroutine nn_wall_model_compute


  subroutine read_npy(fname, arr, n)
    character(len=*), intent(in) :: fname
    real(kind=c_float), intent(out) :: arr(n)
    integer, intent(in) :: n
    integer :: i
    open(unit=77, file=fname, status='old', action='read')
    do i = 1, n
      read(77, *) arr(i)
    end do
    close(77)
  end subroutine read_npy

end module nn_wall_model
